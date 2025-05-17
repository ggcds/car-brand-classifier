# Classificação de Marcas de Carro com MobileNetV2

Projeto desenvolvido como parte da disciplina **Resolução de Problemas com Deep Learning** do MBA em BI & Data Science, Ibmec.  
Aqui utilizamos o modelo pré‑treinado **MobileNetV2** para classificar imagens de 10 marcas de carro.

---

## 🧠 Objetivo

- Baixar automaticamente um grande volume de imagens de 10 marcas populares (Ferrari, Lamborghini, Porsche, BMW, Audi, Mercedes, Tesla, Toyota, Ford e Volkswagen).  
- Organizar estas imagens em **train**, **val** e **test** (70 % / 15 % / 15 %).  
- Treinar o modelo e fazer fine‑tuning de um classificador baseado em MobileNetV2.  
- Avaliar o desempenho e servir predições em imagens individuais.

---

## 🔧 Tecnologias

- Python 3.x  
- TensorFlow / Keras  
- icrawler (GoogleImageCrawler)  
- Matplotlib, NumPy, scikit‑learn  

---

## 📁 Estrutura do Projeto

```
/
├── dataset/
│   ├── train/            # Imagens de treino organizadas por classe
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
│   ├── val/              # Imagens de validação (mesma estrutura acima)
│   └── test/             # Imagens de teste (mesma estrutura acima)
│
├── download_and_split.py # Script Python para baixar e organizar as imagens
├── img_classifier.ipynb  # Notebook de treino, fine‑tuning e avaliação
└── README.md             # Este arquivo
```

---

## ▶️ Como Executar

1. **Clone o repositório**  
   ```bash
   git clone https://github.com/seu-usuario/nome-do-repo.git
   cd nome-do-repo
   ```

2. **Configure o ambiente** (opcional, mas recomendado)  
   ```bash
   python -m venv venv
   # Linux / macOS
   source venv/bin/activate
   # Windows
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Baixe e organize as imagens**  
   ```bash
   python download_and_split.py
   ```

4. **Treine o modelo**  
   - Abra o notebook:  
     ```bash
     jupyter notebook img_classifier.ipynb
     ```
   - Execute as células na ordem:  
     1. Preparação do MobileNetV2 (backbone + cabeça personalizada)  
     2. Compilação, data augmentation e criação dos geradores  
     3. Treinamento inicial (10 épocas)  
     4. Fine‑tuning (5 épocas)

5. **Teste individual de imagens**  
   No final do notebook há a função:
   ```python
   predict_and_show("dataset/test/bmw/000002.jpg")
   ```

---

## 🖼️ Saída Esperada

- **Treino / Validação**:  
  Curvas de acurácia e loss ao longo das épocas, com uso opcional de callbacks (_EarlyStopping_, _ModelCheckpoint_).  
- **Test set**:  
  Métricas finais (loss e accuracy) em `dataset/test/`, matriz de confusão e relatório de classificação.  
- **Predição individual**:  
  Visualização da imagem e label prevista com probabilidade.

---

## 🚀 Possíveis Melhorias

- Aumentar o número de imagens por classe  
- Augmentations avançados (_MixUp_, _CutMix_, _Albumentations_)  
- Testar outras backbones (EfficientNet, ResNet…)  
- Usar _ReduceLROnPlateau_, _CosineDecay_ para agendamento de learning rate  
- Analisar Grad‑CAM para entender onde o modelo focaliza a imagem  

---

**Desenvolvido por Guilherme Caldas**  
MBA em BI & Data Science, Ibmec  
