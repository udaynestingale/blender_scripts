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
    
@app.route('/replaceProduct', methods=['POST']) #POST
def process():
    try:
        print("Received request to replace product")
        json_data = request.json
        product_sku_id = json_data['product_sku_id']
        glb_image_key = json_data['glb_image_key']
        generated_2d_image_key = json_data['generated_2d_image_key']
        all_masks_key = json_data['all_masks_key']
        target_product_mask_key = json_data['target_product_mask_key']
        camera_info = json_data['camera_info']
        lighting_info = json_data['lighting_info']
        replace_product_data = json_data['replace_product_data']

        print("JSON data, product_sku_id :" + product_sku_id + "glb_image_key: " + glb_image_key + ", generated_2d_image_key: "+ generated_2d_image_key + ", all_masks_key: "+ all_masks_key + ", target_product_mask_key: "+ target_product_mask_key + ", camera_info: "+ camera_info + ", lighting_info: "+ lighting_info+ ", replace_product_data: "+ replace_product_data)

        # Run the process in a new thread to handle parallel requests
        process_thread = Thread(target=process_request, args=(product_sku_id, glb_image_key, generated_2d_image_key, all_masks_key, target_product_mask_key, camera_info, lighting_info, replace_product_data))
        process_thread.start()

        return jsonify({"message": "Processing started"}), 202
     
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def process_request(product_sku_id, glb_image_key, generated_2d_image_key, all_masks_key, target_product_mask_key, camera_info, lighting_info, replace_product_data):
    try:
        # Generate unique file paths using UUID
        #request_id = str(uuid.uuid4())
        input_file_path = f'/home/ubuntu/product_replacement_script/input_image.glb'
        blender_script_path = f'/home/ubuntu/product_replacement_script/blender_script_camera_public.py'  # Path to your Blender Python script
        output_dir = '/home/ubuntu/product_replacement_script/generated_files'
        generated_2d_image_local_path = f'{output_dir}/room_render.png'
        all_masks_local_path = f'{output_dir}/mask_all_products.png'
        target_product_mask_local_path = f'{output_dir}/mask_{product_sku_id}.png'
        camera_info = json.dumps(camera_info)
        lighting_info = json.dumps(lighting_info)
        replace_product_data = json.dumps(replace_product_data)
        
        # Download input image from S3
        s3_client.download_file(S3_BUCKET_NAME, glb_image_key, input_file_path)
        print(f"File {glb_image_key} downloaded to {input_file_path}")

        print("Command used:")
        print(f'blender -b -P {blender_script_path} -- {input_file_path} -d {output_dir} --generate-mask --camera-json {camera_info} --lighting-json {lighting_info} --use-environment-map "studio.exr" --use-existing-camera --replace-product {replace_product_data}')

        os.system(f'blender -b -P {blender_script_path} -- {input_file_path} -d {output_dir} --generate-mask --camera-json {camera_info} --lighting-json {lighting_info} --use-environment-map "studio.exr" --use-existing-camera --replace-product {replace_product_data}')
        print("2d image and masks are generated")

        s3_client.upload_file(generated_2d_image_local_path, S3_BUCKET_NAME, generated_2d_image_key)
        s3_client.upload_file(all_masks_local_path, S3_BUCKET_NAME, all_masks_key)
        s3_client.upload_file(target_product_mask_local_path, S3_BUCKET_NAME, target_product_mask_key)
        print(f"File {generated_2d_image_local_path} uploaded to {generated_2d_image_key}")
        print(f"File {all_masks_local_path} uploaded to {all_masks_key}")
        print(f"File {target_product_mask_local_path} uploaded to {target_product_mask_key}")
        print("2d image and masks upload completed")

        # Clean up local files after processing
        os.remove(generated_2d_image_local_path)
        os.remove(all_masks_local_path)
        os.remove(target_product_mask_local_path)

        message_body = {
            "eventType": "ProductReplaced",
            "productSkuId": product_sku_id,
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
        print(f"Error processing request {glb_image_key}: {str(e)}")  
        message_body = {
            "eventType": "ProductReplacementFailed",
            "productSkuId": product_sku_id,
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
    app.run(host='0.0.0.0', port=5020)

