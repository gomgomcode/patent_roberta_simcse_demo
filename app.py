import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
import gradio as gr

# Import our models. The package will take care of downloading the models automatically
tokenizer = AutoTokenizer.from_pretrained("gomgomcode/material_patent_roberta_simcse")
model = AutoModel.from_pretrained("gomgomcode/material_patent_roberta_simcse")

def simcse(text1, text2, text3):
    # Tokenize input texts
    texts = [
        text1,
        text2,
        text3
    ]
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # Get the embeddings
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

    # Calculate cosine similarities
    # Cosine similarities are in [-1, 1]. Higher means more similar
    cosine_sim_0_1 = 1 - cosine(embeddings[0], embeddings[1])
    cosine_sim_0_2 = 1 - cosine(embeddings[0], embeddings[2])
    # return {"cosine similarity":cosine_sim_0_1}, {"cosine similarity":cosine_sim_0_2}
    return {"cosine similarity":cosine_sim_0_1}, cosine_sim_0_1, {"cosine similarity":cosine_sim_0_2}, cosine_sim_0_2

inputs = [
    gr.components.Textbox(lines=5, label="Input Text One"),
    gr.components.Textbox(lines=5, label="Input Text Two"),
    gr.components.Textbox(lines=5, label="Input Text Three")
]

outputs = [
    gr.components.Label(label="Cosine similarity between text one and two"),
    gr.components.Number(label="Cosine similarity between text one and two"),
    gr.components.Label(label="Cosine similarity between text one and three"),
    gr.components.Number(label="Cosine similarity between text one and three"),
]

title = "material_patent_roberta_simcse"
description = "첨단 융합 소재 발굴을 위한 인공지능 지원 플랫폼 개발 프로젝트를 위한 특허 임베딩 모델"
article = "<p style='text-align: center'><a href='https://huggingface.co/gomgomcode/material_patent_roberta_simcse'>첨단 융합 소재 발굴을 위한 인공지능 지원 플랫폼 개발 프로젝트:특허 임베딩 모델</a></p>"
examples = [
    ["반도체 식각 공정",
    "반도체 기판의 상부에 식각 대상막을 형성하는 단계;상기 식각 대상막의 상부에 제1 및 제2 식각 보조막을 적층하는 단계;상기 제2 식각 보조막의 상부에 포토레지스트 패턴을 형성하는 단계;상기 포토레지스트 패턴을 식각 베리어로 이용한 식각 공정으로 상기 제1 및 제2 식각 보조막을 식각하여 제1 및 제2 식각 보조 패턴을 형성하는 단계;상기 제1 식각 보조 패턴의 측벽을 식각하는 단계; 및상기 제1 식각 보조 패턴을 식각 베리어로 이용한 식각 공정으로 상기 식각 대상막을 식각하는 단계를 포함하는 반도체 소자의 패턴 형성방법.",
    "본 발명의 실시예는 플렉서블 표시패널을 두 가지 이상의 형태를 안정적으로 유지하는 쌍안정(bistability)이 가능하도록 구현함으로써 두 가지 이상의 형태로 변경될 수 있는 플렉서블 표시장치에 관한 것이다. 본 발명의 일 실시예에 따른 플렉서블 표시장치는 하부 기판, 플라스틱 필름, 제1 및 제2 형태 변경 수단들을 포함한다. 상기 하부 기판은 적어도 두 가지의 형태를 안정적으로 유지하는 쌍안정이 가능한 물질로 이루어진다. 상기 플라스틱 기판은 상기 하부 기판 상에 배치되며, 화소 어레이가 형성된다. 상기 제1 형태 변경 수단은 상기 하부 기판을 제1 안정 상태로 변형하기 위해 상기 하부 기판의 일면에 배치된다. 상기 제2 형태 변경 수단은 제2 안정 상태로 변형하기 위해 상기 하부 기판의 일면의 반대면에 배치된다."],
    ["배터리 음극재 재활용",
     "[청구항1]\n (1) 배터리에서 분리된 LMOX를 포함하는 양극재를 염소를 포함하는 기체와 염소화 반응시켜 제1 혼합물을 형성하는 단계(S100);\n(2) 상기 제1혼합물을 용매와 접촉시켜 MOx를 분리하고 용매를 포함하는 제2혼합물을 형성하는 단계(S200);\n(3) 상기 제2혼합물을 탄산염과 반응시켜 MCO3를 분리하는 단계(S300); 및\n(4) 상기 MCO3가 분리된 제2혼합물에서 탄산리튬(Li2CO3)을 분리하는 단계(S400); 를 포함하는 이차전지 양극재의 재활용 방법.\n이때 상기 L은 Li(리튬)이고, 상기 M은 Co(코발트), Ni(니켈), Al(알루미늄), Mn(망간)에서 선택되는 1종 이상이며, x는 0.5 내지 2.5의 상수이다",
     "본 발명은 폴리에틸렌테레프탈레이트 폐기물을 글리콜류 물질과 촉매 하에서 에스테르 교환 반응으로 해중합 하는 단량체 형성단계와, 증류 방법으로 단량체 형성단계에서 만들어진 해중합물에 남아있는 글리콜류 물질을 회수하는 제1 회수단계와, 글리콜류 물질이 회수된 해중합물을 단량체와 중합체로 분리 회수하는 제2 회수단계를 포함하여 이루어지며, 제2 회수단계의 세부 구성에 따라 환경오염 최소화, 에너지 절감, BHET 순도 상승, 공정 단순화 효과를 가지는 폴리에틸렌테레프탈레이트의 비스테레프탈레이트 분리 및 회수방법에 관한 것이다."],
    ["메탄가스 정제 방법",
     "[청구항1]\n 이산화탄소 함유 메탄가스로부터 메탄올을 제조하는 방법에 있어서,\n이산화탄소 함유 메탄가스에 포함된 황화수소를 제거하는 제1단계;\n이산화탄소의 함량이 몰수 기준으로 메탄 대비 0.2 내지 0.7 이 되도록, 이산화탄소 일부를 제거하여 플라즈마에 의한 수증기 개질 반응 가스 중에 포함된 이산화탄소의 양을 조절하는 제2단계;\n이산화탄소의 양이 조절된 메탄가스를 플라즈마에 의해 수증기 개질하여 합성가스를 제조하는 제3단계; 및\n제3단계에 의해 제조한 합성가스로부터 메탄올을 제조하는 제4단계\n를 포함하는 것이 특징인 메탄올 제조방법으로서,\n상기 이산화탄소 함유 메탄가스는 이산화탄소 및 메탄의 함유량이 건조 가스 몰수 기준으로 각각 30% 및 40% 이상 포함하는 가스로 실록산을 포함하고,\n상기 플라즈마에 의한 수증기 개질 반응을 통해 합성가스를 제조하는 제3단계에서 실록산을 실리카 입자로 전환하며, 암모니아 및 휘발성 유기화합물은 열분해되는 것이 특징인, 제조방법.",
     "[청구항1]\n (1) 배터리에서 분리된 LMOX를 포함하는 양극재를 염소를 포함하는 기체와 염소화 반응시켜 제1 혼합물을 형성하는 단계(S100);\n(2) 상기 제1혼합물을 용매와 접촉시켜 MOx를 분리하고 용매를 포함하는 제2혼합물을 형성하는 단계(S200);\n(3) 상기 제2혼합물을 탄산염과 반응시켜 MCO3를 분리하는 단계(S300); 및\n(4) 상기 MCO3가 분리된 제2혼합물에서 탄산리튬(Li2CO3)을 분리하는 단계(S400); 를 포함하는 이차전지 양극재의 재활용 방법.\n이때 상기 L은 Li(리튬)이고, 상기 M은 Co(코발트), Ni(니켈), Al(알루미늄), Mn(망간)에서 선택되는 1종 이상이며, x는 0.5 내지 2.5의 상수이다"]
]

gr.Interface(simcse, inputs, outputs, title=title, description=description, article=article, examples=examples).launch(share=True)
