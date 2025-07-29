from flask import Flask, request, jsonify
import boto3
import os
from threading import Thread
import json

app = Flask(__name__)

S3_BUCKET_NAME = 'nestingale-dev-digital-assets'
S3_REGION = 'us-east-1'

s3_client = boto3.client('s3', region_name=S3_REGION)
sqs = boto3.client('sqs', region_name=S3_REGION)
queue_url = "https://sqs.us-east-1.amazonaws.com/311504593279/EmailMarketing"

@app.route('/healthCheck', methods=['GET'])
def health():
    return jsonify({"message": "Success"}), 200
    
@app.route('/processGlb', methods=['POST']) #POST
def process():
    try:
        json_data = request.json
        product_type = json_data['product_type']
        product_image_s3_path = json_data['product_image_s3_path']
        product_image_s3_path2 = json_data['product_image_s3_path2']
        product_sku_id = json_data['product_sku_id']
        output_s3_file_key = json_data['output_s3_file_key']
        print("JSON data, product_type: " + json_data['product_type'] + ", product_sku_id: "+ json_data['product_sku_id']  + ", product_image_s3_path: "+ json_data['product_image_s3_path'] + ", product_image_s3_path2: "+ json_data['product_image_s3_path2'])
         # Run the process in a new thread to handle parallel requests
        process_thread = Thread(target=process_request, args=(product_type, product_sku_id, output_s3_file_key,  product_image_s3_path, product_image_s3_path2))
        process_thread.start()
        return jsonify({"message": "Processing started"}), 202
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

def process_request(product_type, product_sku_id, output_s3_file_key, product_image_s3_path, product_image_s3_path2):
    try:
        # Generate unique file paths using UUID
        #request_id = str(uuid.uuid4())
        input_file_path = f'/home/ubuntu/product_2d_to_3d/input_image_{product_sku_id}.png'
        blender_script_path = '/home/ubuntu/product_2d_to_3d/create_Rug_or_Pillow_GLB_public.py'
        #base_rug_glb = '/home/ubuntu/product_2d_to_3d/Rug_Rectangular_Base_10x13.glb'
        #base_pillow_glb = '/home/ubuntu/product_2d_to_3d/Pillow_Square_Base_40x40.glb'

        # Download input image from S3
        s3_client.download_file(S3_BUCKET_NAME, product_image_s3_path, input_file_path)
        print(f"File {product_image_s3_path} downloaded to {input_file_path}")
        
        if product_type == "pillow":
            input_file_path2 = f'/home/ubuntu/product_2d_to_3d/input_image2_{product_sku_id}.png'
            s3_client.download_file(S3_BUCKET_NAME, product_image_s3_path2, input_file_path2)
            print(f"File {product_image_s3_path2} downloaded to {input_file_path2}")

        # Run Blender Python script
        if product_type == "rug":
            print(f'blender -b -P {blender_script_path} -- --{product_type} {input_file_path}')
            os.system(f'blender -b -P {blender_script_path} -- --{product_type} {input_file_path}')
        elif product_type == "pillow":
            print(f'blender -b -P {blender_script_path} -- --{product_type} {input_file_path} {input_file_path2}')
            os.system(f'blender -b -P {blender_script_path} -- --{product_type} {input_file_path} {input_file_path2}')
        else:
            print(f"Unsupported product type: {product_type}")

        print(f"Blender processing complete for {input_file_path}")

        output_file_path = f'/home/ubuntu/product_2d_to_3d/output.glb'
        # Upload output file back to S3
        s3_client.upload_file(output_file_path, S3_BUCKET_NAME, output_s3_file_key)
        print(f"File {output_file_path} uploaded to {output_s3_file_key}")

        # Clean up local files after processing
        os.remove(input_file_path)
        os.remove(output_file_path)

        message_body = {
            "eventType": "twodToThreedFileCreated",
            "projectId": product_sku_id,
        }

        # Convert the dictionary to a JSON string
        message_body_json = json.dumps(message_body)

        # Send message to SQS queue
        response = sqs.send_message(
            QueueUrl=queue_url,
            MessageBody=message_body_json
        )
        print("Message sent to SQS:", response)


    except Exception as e:
        print(f"Error processing request for SkuId :  {product_sku_id}: {str(e)}")  
        message_body = {
            "eventType": "twodToThreedFileFailed",
            "projectId": product_sku_id,
        }

        # Convert the dictionary to a JSON string
        message_body_json = json.dumps(message_body)

        # Send message to SQS queue
        response = sqs.send_message(
            QueueUrl=queue_url,
            MessageBody=message_body_json
        )
        print("Message sent to SQS:", response)
  

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5018)
