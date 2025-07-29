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
    
@app.route('/generatePhotoRealisticView', methods=['POST']) #POST
def process():
    try:
        json_data = request.json
        template_id = json_data['template_id']
        glb_image_key = json_data['glb_image_key']
        generated_2d_image_key = json_data['generated_2d_image_key']
        all_masks_key = json_data['all_masks_key']
        camera_info = json_data['camera_info']
        lighting_info = json_data['lighting_info']

        print("JSON data, template_id :" + template_id + "glb_image_key: " + glb_image_key + ", generated_2d_image_key: "+ generated_2d_image_key + ", all_masks_key: "+ all_masks_key + ", camera_info: "+ camera_info + ", lighting_info: "+ lighting_info)

        # Run the process in a new thread to handle parallel requests
        process_thread = Thread(target=process_request, args=(template_id, glb_image_key, generated_2d_image_key, all_masks_key, camera_info, lighting_info))
        process_thread.start()

        return jsonify({"message": "Processing started"}), 202
     
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

def process_request(template_id, glb_image_key, generated_2d_image_key, all_masks_key, camera_info, lighting_info):
    try:
        # Generate unique file paths using UUID
        #request_id = str(uuid.uuid4())
        input_file_path = f'/home/ubuntu/photo_realistic_view_generation/input_image.glb'
        blender_script_path = f'/home/ubuntu/photo_realistic_view_generation/blender_script.py'  # Path to your Blender Python script
        output_dir = '/home/ubuntu/photo_realistic_view_generation/generated_files';
        generated_2d_image_local_path = f'{output_dir}/room_render.png'
        all_masks_local_path = f'{output_dir}/mask_all_products.png'
        camera_info = json.dumps(camera_info)
        lighting_info = json.dumps(lighting_info)
        
        # Download input image from S3
        s3_client.download_file(S3_BUCKET_NAME, glb_image_key, input_file_path)
        print(f"File {glb_image_key} downloaded to {input_file_path}")

        print("Command used:")
        print(f'blender -b -P {blender_script_path} -- {input_file_path} -d {output_dir} --generate-mask --combined-mask-only -r 1920 --camera-json {camera_info} --lighting-json {lighting_info}')

        os.system(f'blender -b -P {blender_script_path} -- {input_file_path} -d {output_dir} --generate-mask --combined-mask-only -r 1920 --camera-json {camera_info} --lighting-json {lighting_info}')
        print("2d image and masks are generated")

        s3_client.upload_file(generated_2d_image_local_path, S3_BUCKET_NAME, generated_2d_image_key)
        s3_client.upload_file(all_masks_local_path, S3_BUCKET_NAME, all_masks_key)
        print(f"File {generated_2d_image_local_path} uploaded to {generated_2d_image_key}")
        print(f"File {all_masks_local_path} uploaded to {all_masks_key}")
        print("2d image and masks upload completed")

        # Clean up local files after processing
        os.remove(generated_2d_image_local_path)
        os.remove(all_masks_local_path)

        message_body = {
            "eventType": "photoRealisticViewGenerated",
            "templateId": template_id,
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
            "eventType": "photoRealisticViewFailed",
            "templateId": template_id,
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
    app.run(host='0.0.0.0', port=5016)

