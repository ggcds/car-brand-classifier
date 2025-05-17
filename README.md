# Classificação de Marcas de Carro com MobileNetV2

Projeto desenvolvido como parte da disciplina **Resolução de Problemas com Deep Learning** do MBA em BI & Data Science, Ibmec.  
Aqui utilizamos o modelo pré‑treinado **MobileNetV2** para classificar imagens de 10 marcas de carro.

---

## 🧠 Objetivo

- Baixar automaticamente um grande volume de imagens de 10 marcas populares (Ferrari, Lamborghini, Porsche, BMW, Audi, Mercedes, Tesla, Toyota, Ford e Volkswagen).  
- Organizar estas imagens em **treino**, **validação** e **teste** (70/15/15).  
- Treinar e fazer fine‑tuning de um classificador MobileNetV2.  
- Avaliar o desempenho e servir predições em imagens avulsas.

---

## 🔧 Tecnologias

- Python 3.x  
- TensorFlow / Keras  
- icrawler (GoogleImageCrawler)  
- Matplotlib, NumPy, scikit‑learn  

---

## 📁 Estrutura do Projeto

\`\`\`
/
├── dataset/  
│   ├── train/               # Imagens de treino organizadas por classe  
│   │   ├── audi/  
│   │   ├── bmw/  
│   │   ├── ferrari/  
│   │   ├── ford/  
│   │   ├── lamborghini/  
│   │   ├── mercedes/  
│   │   ├── porsche/  
│   │   ├── tesla/  
│   │   ├── toyota/  
│   │   └── volkswagen/  
│   ├── val/                 # Imagens de validação  
│   │   └── (mesma estrutura acima)  
│   └── test/                # Imagens de teste  
│       └── (mesma estrutura acima)  
│
├── download_and_split.py    # Script Python para baixar e organizar as imagens  
├── img_classifier.ipynb     # Notebook de treino, fine‑tuning e avaliação  
└── README.md                # Este arquivo  
\`\`\`

---

## ▶️ Como Executar

1. **Clone o repositório**  
   \`\`\`bash
   git clone https://github.com/seu-usuario/nome-do-repo.git
   cd nome-do-repo
   \`\`\`

2. **Configure ambiente** (opcional mas recomendado)  
   \`\`\`bash
   python -m venv vvenv
   source venv/bin/activate        # Linux / macOS  
   venv\Scripts\activate         # Windows
   pip install -r requirements.txt
   \`\`\`

3. **Baixe e organize imagens**  
   \`\`\`bash
   python download_and_split.py
   \`\`\`

4. **Treine o modelo**  
   - Abra \`img_classifier.ipynb\` no Jupyter Notebook:  
     \`\`\`bash
     jupyter notebook img_classifier.ipynb
     \`\`\`
   - Execute as células de:  
     1. Preparação do MobileNetV2 (backbone + cabeça personalizada)  
     2. Compilação, data augmentation e geradores de dados  
     3. Treinamento inicial (10 épocas)  
     4. Fine‑tuning (5 épocas)

5. **Teste individual de imagens**  
   No final do notebook há a função:  
   \`\`\`python
   predict_and_show("dataset/test/bmw/000002.jpg")
   \`\`\`

---

## 🖼️ O que esperar

- **Treino / Validação**:  
  Métricas de acurácia e loss ao longo das épocas, com callbacks opcionais de _EarlyStopping_ e _ModelCheckpoint_ para salvar o melhor modelo.  
- **Test set**:  
  Avaliação final (loss e accuracy) em \`dataset/test/\`, geração de matriz de confusão e relatório de classificação.  
- **Predição individual**:  
  Visualização da imagem + label prevista e confiança.

---

## 🚀 Possíveis Melhorias

- Aumentar / filtrar mais imagens para cada classe  
- Augmentations avançados (MixUp, CutMix, Albumentations)  
- Testar outras backbones (EfficientNet, ResNet)  
- Agendar callbacks de _ReduceLROnPlateau_, _CosineDecay_  
- Analisar Grad‑CAM para entender o foco do modelo  

---

**Desenvolvido por Guilherme Caldas**  
MBA em BI & Data Science, Ibmec
