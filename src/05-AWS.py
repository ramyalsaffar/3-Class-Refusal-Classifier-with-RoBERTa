# AWS Configuration & Secrets Handler
#--------------------------------------
# This file contains:
# - AWS configuration for cloud deployment
# - AWS Secrets Manager handler for API keys
###############################################################################


# =============================================================================
# AWS CONFIGURATION
# =============================================================================

# AWS Configuration (Optional)
#-----------------------------
AWS_CONFIG = {
    'enabled': IS_AWS,
    'region': os.getenv('AWS_REGION', 'us-east-1'),
    's3_bucket': os.getenv('S3_BUCKET_NAME', 'refusal-classifier-results'),
    's3_results_prefix': 'runs/',
    's3_logs_prefix': 'logs/',

    # AWS Secrets Manager keys
    'secrets': {
        'openai': os.getenv('SECRETS_OPENAI_KEY_NAME', 'refusal-classifier/openai-api-key'),
        'anthropic': os.getenv('SECRETS_ANTHROPIC_KEY_NAME', 'refusal-classifier/anthropic-api-key'),
        'google': os.getenv('SECRETS_GOOGLE_KEY_NAME', 'refusal-classifier/google-api-key')
    },

    # EC2 Configuration
    'ec2_instance_type': os.getenv('EC2_INSTANCE_TYPE', 'g4dn.xlarge'),  # GPU instance
    'ec2_security_group': 'refusal-classifier-sg',
    'iam_role_name': 'refusal-classifier-ec2-role'
}


# =============================================================================
# AWS SECRETS MANAGER HANDLER
# =============================================================================

class SecretsHandler:
    """Handle AWS Secrets Manager operations"""

    def __init__(self, region='us-east-1'):
        """
        Initialize Secrets Manager handler

        Args:
            region: AWS region (default: us-east-1)
        """
        self.region = region
        self.client = boto3.client('secretsmanager', region_name=region)

    def __enter__(self):
        """Context manager entry - FIX: Support 'with' statement for proper cleanup"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - FIX: Ensure client is closed"""
        self.close()
        return False

    def close(self):
        """FIX: Explicitly close boto3 client to release resources"""
        if hasattr(self, 'client') and self.client is not None:
            self.client.close()
            self.client = None


    def get_secret(self, secret_name):
        """
        Retrieve secret from AWS Secrets Manager

        Args:
            secret_name: Name of the secret

        Returns:
            Secret value (string or dict)
        """
        try:
            response = self.client.get_secret_value(SecretId=secret_name)

            # Secrets can be string or binary
            if 'SecretString' in response:
                secret = response['SecretString']
                # Try to parse as JSON
                try:
                    return json.loads(secret)
                except json.JSONDecodeError:
                    return secret
            else:
                # Binary secret
                return response['SecretBinary']

        except ClientError as e:
            error_code = e.response['Error']['Code']

            if error_code == 'ResourceNotFoundException':
                print(f"❌ Secret '{secret_name}' not found")
            elif error_code == 'InvalidRequestException':
                print(f"❌ Invalid request for secret '{secret_name}'")
            elif error_code == 'InvalidParameterException':
                print(f"❌ Invalid parameter for secret '{secret_name}'")
            elif error_code == 'DecryptionFailure':
                print(f"❌ Failed to decrypt secret '{secret_name}'")
            elif error_code == 'InternalServiceError':
                print(f"❌ AWS internal error retrieving secret '{secret_name}'")
            else:
                print(f"❌ Error retrieving secret: {e}")

            raise


    def get_api_key(self, secret_name, key_field='api_key'):
        """
        Retrieve API key from Secrets Manager

        Args:
            secret_name: Name of the secret containing API key
            key_field: Field name in JSON secret (default: 'api_key')

        Returns:
            API key string
        """
        print(f"🔐 Retrieving API key from Secrets Manager ({secret_name})...")

        secret = self.get_secret(secret_name)

        # Handle different secret formats
        if isinstance(secret, dict):
            # If stored as JSON with key field
            api_key = secret.get(key_field) or secret.get(key_field.upper())
            if not api_key:
                raise ValueError(f"Secret '{secret_name}' does not contain field '{key_field}'")
            return api_key
        else:
            # If stored as plain string
            return secret


    def create_secret(self, secret_name, secret_value, description=''):
        """
        Create a new secret in Secrets Manager

        Args:
            secret_name: Name for the secret
            secret_value: Secret value (string or dict)
            description: Optional description

        Returns:
            Secret ARN
        """
        try:
            # Convert dict to JSON string
            if isinstance(secret_value, dict):
                secret_string = json.dumps(secret_value)
            else:
                secret_string = str(secret_value)

            response = self.client.create_secret(
                Name=secret_name,
                Description=description,
                SecretString=secret_string
            )

            print(f"✅ Created secret '{secret_name}'")
            return response['ARN']

        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceExistsException':
                print(f"⚠️ Secret '{secret_name}' already exists")
                # Update instead
                return self.update_secret(secret_name, secret_value)
            else:
                print(f"❌ Error creating secret: {e}")
                raise


    def update_secret(self, secret_name, secret_value):
        """
        Update existing secret

        Args:
            secret_name: Name of the secret
            secret_value: New secret value

        Returns:
            Secret ARN
        """
        try:
            # Convert dict to JSON string
            if isinstance(secret_value, dict):
                secret_string = json.dumps(secret_value)
            else:
                secret_string = str(secret_value)

            response = self.client.update_secret(
                SecretId=secret_name,
                SecretString=secret_string
            )

            print(f"✅ Updated secret '{secret_name}'")
            return response['ARN']

        except ClientError as e:
            print(f"❌ Error updating secret: {e}")
            raise


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual RoBERTa Classifiers: 3-Class Refusal Taxonomy & Binary Jailbreak Detection
Created on October 28, 2025
@author: ramyalsaffar
"""
