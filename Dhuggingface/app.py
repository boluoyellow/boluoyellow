import os
from flask import Flask, render_template, request
from huggingface_hub import InferenceClient

app = Flask(__name__)

MODEL_ID = "XLabs-AI/flux-RealismLora"

# 从环境变量读取 Hugging Face Token
token = os.environ["HF_TOKEN"]

# 创建 Hugging Face 推理客户端
client = InferenceClient(
    provider="fal-ai",
    api_key=token
)

@app.route("/", methods=["GET", "POST"])
def index():
    # 只要之前已经生成过 result.png，就继续显示它
    image_generated = os.path.exists("static/result.png")

    message = ""

    # 默认显示我们刚才成功生成时使用的提示词
    prompt = """A highly realistic photograph of a rainy modern city street at night, wet pavement reflecting warm street lights and shop signs, pedestrians holding umbrellas, natural perspective, realistic buildings, detailed textures, soft cinematic lighting, photorealistic, real-world photography"""

    if request.method == "POST":
        prompt = request.form.get("prompt")

        try:
            print("=" * 50)
            print("收到前端提示词：", prompt)
            print("正在调用 Hugging Face API...")
            print("调用模型：", MODEL_ID)

            image = client.text_to_image(
                prompt,
                model=MODEL_ID
            )

            image.save("static/result.png")

            print("API 调用成功！")
            print("图像已保存：static/result.png")
            print("=" * 50)

            image_generated = True
            message = "图像生成成功！"

        except Exception as e:
            print("API 调用失败：", e)
            message = "生成失败：" + str(e)

    return render_template(
        "index.html",
        image_generated=image_generated,
        message=message,
        prompt=prompt
    )

if __name__ == "__main__":
    app.run(debug=True)