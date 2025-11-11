# AWS Configuration & Handlers
#-------------------------------
# This file contains:
# - AWS configuration for cloud deployment
# - AWS Secrets Manager handler for API keys
# - S3 handler for results storage
# All imports are in 01-Imports.py
###############################################################################


# =============================================================================
# AWS CONFIGURATION
# =============================================================================

# AWS Configuration
#------------------
# Cloud deployment settings for AWS/Docker environments
AWS_CONFIG = {
    'enabled': IS_AWS,
    'region': os.getenv('AWS_REGION', 'us-east-1'),
    's3_bucket': os.getenv('S3_BUCKET_NAME', 'refusal-classifier-results'),
    's3_results_prefix': 'runs/',
    's3_logs_prefix': 'logs/',
    's3_checkpoints_prefix': 'checkpoints/',
    
    # AWS Secrets Manager keys
    'secrets': {
        'openai': os.getenv('SECRETS_OPENAI_KEY_NAME', 'refusal-classifier/openai-api-key'),
        'anthropic': os.getenv('SECRETS_ANTHROPIC_KEY_NAME', 'refusal-classifier/anthropic-api-key'),
        'google': os.getenv('SECRETS_GOOGLE_KEY_NAME', 'refusal-classifier/google-api-key'),
        'default_key_field': 'api_key'  # Default field name in JSON secrets
    },
    
    # EC2 Configuration
    'ec2_instance_type': os.getenv('EC2_INSTANCE_TYPE', 'g4dn.xlarge'),  # GPU instance
    'ec2_security_group': 'refusal-classifier-sg',
    'iam_role_name': 'refusal-classifier-ec2-role',
    
    # Resource limits
    'max_s3_file_size_mb': 100,  # Maximum file size for S3 uploads
    'max_secrets_cache_age_seconds': 3600  # Cache secrets for 1 hour
}


# =============================================================================
# AWS AVAILABILITY CHECK
# =============================================================================

def check_aws_available() -> bool:
    """Check if AWS SDK (boto3) is available."""
    if not AWS_AVAILABLE:
        if EXPERIMENT_CONFIG.get('verbose', True):
            print("⚠️  AWS features not available - boto3 not installed")
            print("   Install with: pip install boto3")
        return False
    return True


# =============================================================================
# AWS SECRETS MANAGER HANDLER
# =============================================================================

class SecretsHandler:
    """
    Handle AWS Secrets Manager operations.
    
    IMPORTANT: Uses AWS_CONFIG for all settings, no hardcoded values!
    """
    
    def __init__(self, region: str = None, verbose: bool = None):
        """
        Initialize Secrets Manager handler.
        
        Args:
            region: AWS region (default: from AWS_CONFIG)
            verbose: Override verbosity (default: from EXPERIMENT_CONFIG)
        """
        if not check_aws_available():
            raise ImportError("boto3 is required for AWS features")
            
        # Use config values - NO HARDCODING!
        self.region = region or AWS_CONFIG['region']
        self.verbose = verbose if verbose is not None else EXPERIMENT_CONFIG.get('verbose', True)
        
        # Initialize client
        self.client = boto3.client('secretsmanager', region_name=self.region)
        
        # Cache for retrieved secrets
        self._cache = {}
        self._cache_timestamps = {}
        self.max_cache_age = AWS_CONFIG.get('max_secrets_cache_age_seconds', 3600)
        
        if self.verbose:
            print_banner(f"AWS SECRETS MANAGER", width=60, char="-")
            print(f"  Region: {self.region}")
            print(f"  Cache TTL: {self.max_cache_age}s")
            print("-" * 60)
    
    def __enter__(self):
        """Context manager entry - Support 'with' statement."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - Ensure client is closed."""
        self.close()
        return False
    
    def close(self) -> None:
        """Explicitly close boto3 client to release resources."""
        if hasattr(self, 'client') and self.client is not None:
            self.client.close()
            self.client = None
            if self.verbose:
                print("🔒 Secrets Manager client closed")
    
    def _is_cache_valid(self, secret_name: str) -> bool:
        """Check if cached secret is still valid."""
        if secret_name not in self._cache:
            return False
        
        age = time.time() - self._cache_timestamps.get(secret_name, 0)
        return age < self.max_cache_age
    
    def get_secret(self, secret_name: str, use_cache: bool = True) -> Union[str, Dict, bytes]:
        """
        Retrieve secret from AWS Secrets Manager with caching.
        
        Args:
            secret_name: Name of the secret
            use_cache: Use cached value if available (default: True)
        
        Returns:
            Secret value (string or dict)
        """
        # Check cache first
        if use_cache and self._is_cache_valid(secret_name):
            if self.verbose:
                print(f"📦 Using cached secret: {secret_name}")
            return self._cache[secret_name]
        
        try:
            response = self.client.get_secret_value(SecretId=secret_name)
            
            # Parse secret
            if 'SecretString' in response:
                secret = response['SecretString']
                # Try to parse as JSON
                try:
                    secret = json.loads(secret)
                except json.JSONDecodeError:
                    pass  # Keep as string
            else:
                # Binary secret
                secret = response['SecretBinary']
            
            # Cache the secret
            self._cache[secret_name] = secret
            self._cache_timestamps[secret_name] = time.time()
            
            if self.verbose:
                print(f"✅ Retrieved secret: {secret_name}")
            
            return secret
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            
            error_messages = {
                'ResourceNotFoundException': f"Secret '{secret_name}' not found",
                'InvalidRequestException': f"Invalid request for secret '{secret_name}'",
                'InvalidParameterException': f"Invalid parameter for secret '{secret_name}'",
                'DecryptionFailure': f"Failed to decrypt secret '{secret_name}'",
                'InternalServiceError': f"AWS internal error retrieving secret '{secret_name}'"
            }
            
            error_msg = error_messages.get(error_code, f"Error retrieving secret: {e}")
            print(f"❌ {error_msg}")
            raise
    
    def get_api_key(self, secret_name: str, key_field: str = None) -> str:
        """
        Retrieve API key from Secrets Manager.
        
        Args:
            secret_name: Name of the secret containing API key
            key_field: Field name in JSON secret (default: from AWS_CONFIG)
        
        Returns:
            API key string
        """
        # Use config value - NO HARDCODING!
        key_field = key_field or AWS_CONFIG['secrets'].get('default_key_field', 'api_key')
        
        if self.verbose:
            print(f"🔐 Retrieving API key from: {secret_name}")
        
        secret = self.get_secret(secret_name)
        
        # Handle different secret formats
        if isinstance(secret, dict):
            # Try multiple field name variations
            for field in [key_field, key_field.upper(), key_field.lower(), 'key', 'api_key', 'apiKey']:
                if field in secret:
                    return secret[field]
            
            raise ValueError(f"Secret '{secret_name}' does not contain field '{key_field}'")
        else:
            # Plain string secret
            return secret
    
    def create_secret(self, secret_name: str, secret_value: Union[str, Dict], description: str = '') -> str:
        """
        Create a new secret in Secrets Manager.
        
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
                Description=description or f"Created by {EXPERIMENT_CONFIG.get('experiment_name', 'RefusalClassifier')}",
                SecretString=secret_string
            )
            
            if self.verbose:
                print(f"✅ Created secret: {secret_name}")
            
            # Invalidate cache
            self._cache.pop(secret_name, None)
            
            return response['ARN']
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceExistsException':
                if self.verbose:
                    print(f"⚠️  Secret already exists, updating: {secret_name}")
                return self.update_secret(secret_name, secret_value)
            else:
                print(f"❌ Error creating secret: {e}")
                raise
    
    def update_secret(self, secret_name: str, secret_value: Union[str, Dict]) -> str:
        """
        Update existing secret.
        
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
            
            if self.verbose:
                print(f"✅ Updated secret: {secret_name}")
            
            # Invalidate cache
            self._cache.pop(secret_name, None)
            
            return response['ARN']
            
        except ClientError as e:
            print(f"❌ Error updating secret: {e}")
            raise
    
    def list_secrets(self, prefix: str = None) -> List[str]:
        """
        List all secrets, optionally filtered by prefix.
        
        Args:
            prefix: Optional prefix to filter secrets
        
        Returns:
            List of secret names
        """
        try:
            paginator = self.client.get_paginator('list_secrets')
            secret_names = []
            
            for page in paginator.paginate():
                for secret in page['SecretList']:
                    name = secret['Name']
                    if prefix is None or name.startswith(prefix):
                        secret_names.append(name)
            
            if self.verbose and secret_names:
                print(f"📋 Found {len(secret_names)} secret(s)")
            
            return secret_names
            
        except ClientError as e:
            print(f"❌ Error listing secrets: {e}")
            raise


# =============================================================================
# S3 HANDLER
# =============================================================================

class S3Handler:
    """
    Handle S3 operations for results and checkpoints.
    
    IMPORTANT: Uses AWS_CONFIG for all settings!
    """
    
    def __init__(self, bucket: str = None, region: str = None, verbose: bool = None):
        """
        Initialize S3 handler.
        
        Args:
            bucket: S3 bucket name (default: from AWS_CONFIG)
            region: AWS region (default: from AWS_CONFIG)
            verbose: Override verbosity (default: from EXPERIMENT_CONFIG)
        """
        if not check_aws_available():
            raise ImportError("boto3 is required for AWS features")
        
        # Use config values - NO HARDCODING!
        self.bucket = bucket or AWS_CONFIG['s3_bucket']
        self.region = region or AWS_CONFIG['region']
        self.verbose = verbose if verbose is not None else EXPERIMENT_CONFIG.get('verbose', True)
        
        # Initialize client
        self.client = boto3.client('s3', region_name=self.region)
        
        # Prefixes from config
        self.results_prefix = AWS_CONFIG.get('s3_results_prefix', 'runs/')
        self.logs_prefix = AWS_CONFIG.get('s3_logs_prefix', 'logs/')
        self.checkpoints_prefix = AWS_CONFIG.get('s3_checkpoints_prefix', 'checkpoints/')
        
        if self.verbose:
            print_banner("AWS S3 HANDLER", width=60, char="-")
            print(f"  Bucket: {self.bucket}")
            print(f"  Region: {self.region}")
            print("-" * 60)
    
    def upload_file(self, local_path: str, s3_key: str = None) -> str:
        """
        Upload file to S3.
        
        Args:
            local_path: Local file path
            s3_key: S3 object key (default: auto-generate from filename)
        
        Returns:
            S3 URI
        """
        # Check file size
        file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
        max_size = AWS_CONFIG.get('max_s3_file_size_mb', 100)
        
        if file_size_mb > max_size:
            print(f"⚠️  File too large ({file_size_mb:.1f}MB > {max_size}MB)")
            return None
        
        # Generate S3 key if not provided
        if s3_key is None:
            filename = os.path.basename(local_path)
            timestamp = get_timestamp('file')
            s3_key = f"{self.results_prefix}{timestamp}/{filename}"
        
        try:
            self.client.upload_file(local_path, self.bucket, s3_key)
            s3_uri = f"s3://{self.bucket}/{s3_key}"
            
            if self.verbose:
                print(f"✅ Uploaded to S3: {s3_uri}")
                print(f"   Size: {file_size_mb:.2f}MB")
            
            return s3_uri
            
        except ClientError as e:
            print(f"❌ S3 upload failed: {e}")
            return None
    
    def download_file(self, s3_key: str, local_path: str = None) -> str:
        """
        Download file from S3.
        
        Args:
            s3_key: S3 object key
            local_path: Local destination (default: current directory)
        
        Returns:
            Local file path
        """
        if local_path is None:
            local_path = os.path.basename(s3_key)
        
        try:
            self.client.download_file(self.bucket, s3_key, local_path)
            
            if self.verbose:
                file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
                print(f"✅ Downloaded from S3: {local_path}")
                print(f"   Size: {file_size_mb:.2f}MB")
            
            return local_path
            
        except ClientError as e:
            print(f"❌ S3 download failed: {e}")
            return None
    
    def list_objects(self, prefix: str = None, max_results: int = 100) -> List[str]:
        """
        List objects in S3 bucket.
        
        Args:
            prefix: Prefix to filter objects (default: results prefix)
            max_results: Maximum number of results
        
        Returns:
            List of S3 object keys
        """
        prefix = prefix or self.results_prefix
        
        try:
            response = self.client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix,
                MaxKeys=max_results
            )
            
            if 'Contents' not in response:
                return []
            
            objects = [obj['Key'] for obj in response['Contents']]
            
            if self.verbose and objects:
                print(f"📋 Found {len(objects)} object(s) in S3")
            
            return objects
            
        except ClientError as e:
            print(f"❌ S3 list failed: {e}")
            return []
    
    def upload_checkpoint(self, checkpoint_path: str) -> str:
        """
        Upload checkpoint to S3 with proper prefix.
        
        Args:
            checkpoint_path: Local checkpoint file path
        
        Returns:
            S3 URI
        """
        filename = os.path.basename(checkpoint_path)
        s3_key = f"{self.checkpoints_prefix}{filename}"
        return self.upload_file(checkpoint_path, s3_key)
    
    def download_latest_checkpoint(self, operation_name: str, local_dir: str = None) -> str:
        """
        Download the latest checkpoint for an operation.
        
        Args:
            operation_name: Operation name (e.g., 'labeling')
            local_dir: Local directory to save (default: current)
        
        Returns:
            Local file path or None
        """
        prefix = f"{self.checkpoints_prefix}checkpoint_{operation_name}_"
        checkpoints = self.list_objects(prefix)
        
        if not checkpoints:
            if self.verbose:
                print(f"ℹ️  No checkpoints found for: {operation_name}")
            return None
        
        # Get most recent (S3 keys include timestamp)
        latest = sorted(checkpoints)[-1]
        
        local_path = os.path.join(local_dir or '.', os.path.basename(latest))
        return self.download_file(latest, local_path)


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual RoBERTa Classifiers: 3-Class Refusal Taxonomy & Binary Jailbreak Detection
Created on October 28, 2025
@author: ramyalsaffar
"""
