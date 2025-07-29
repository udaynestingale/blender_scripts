from flask import Flask, request, jsonify
import boto3
import os

app = Flask(__name__)

S3_BUCKET_NAME = 'nestingale-dev-product-3d-assets'
S3_REGION = 'us-east-1'

s3_client = boto3.client('s3', region_name=S3_REGION)

@app.route('/healthCheck', methods=['GET'])
def health():
    return jsonify({"message": "Success"}), 200

@app.route('/processGlb', methods=['POST']) #POST
def process():
    try:
        # Get the input and output file paths from the request
        # input_file_key = request.form['input_file_key']  # 'upload3DmodelData/testEC2.usdz'
        # output_file_key = request.form['output_file_key']  # 'upload3DmodelData/output2_ec2.gltf'

        # print("input_file_key: " + input_file_key + ", output_file_key: "+ output_file_key)
        json_data = request.json
        input_file_key = json_data['input_file_key']
        output_file_key = json_data['output_file_key']


        print("JSON data, input_file_key: " + json_data['input_file_key'] + ", output_file_key: "+ json_data['output_file_key'])

        # Download input image from S3
        print("before file downlaod")

        input_file_path = f'/home/ubuntu/blender_with_furniture_scripts/input_image.usdz'
        print("file path, S3_BUCKET_NAME, input_file_key:" + input_file_path, S3_BUCKET_NAME, input_file_key )
        s3_client.download_file(S3_BUCKET_NAME, input_file_key, input_file_path)
        print("file download completed")

        # Run Blender Python script
        blender_script_path = '/home/ubuntu/blender_with_furniture_scripts/blender_with_furniture.py'  # Path to your Blender Python script
        output_file_path = '/home/ubuntu/blender_with_furniture_scripts/output_image.glb'
        os.system(f'blender -b -P {blender_script_path} -- {input_file_path} {output_file_path}')
        print("file conversion completed")
        # Upload output file back to S3

        s3_client.upload_file(output_file_path, S3_BUCKET_NAME, output_file_key)
        print("file upload completed")
        return jsonify({"message": "Image processed and uploaded successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)