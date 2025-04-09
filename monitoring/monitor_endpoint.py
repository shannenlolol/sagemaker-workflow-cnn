import boto3

sm = boto3.client('sagemaker')

response = sm.describe_endpoint(EndpointName='your-endpoint-name')
print(f"Endpoint Status: {response['EndpointStatus']}")
if response['EndpointStatus'] != 'InService':
    print("Warning: Endpoint not available or unhealthy.")
