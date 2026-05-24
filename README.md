# 🧠 Stroke Prediction & Imbalanced Data Handling via SMOTE

[🇹🇷 **Turkish Version Below** / Türkçe Versiyon Aşağıda](#-inme-tahmini--dengesiz-veri-işleme-smote-ile)

---

## 📌 English Version

### 🎯 Project Overview

An end-to-end machine learning pipeline for **stroke prediction** that effectively handles **class imbalance** in medical datasets using SMOTE (Synthetic Minority Over-sampling Technique). This project demonstrates how to build robust diagnostic models that prioritize detecting minority cases (stroke patients) rather than achieving misleadingly high accuracy scores.

**Key Problem Solved:** Medical datasets often suffer from severe class imbalance (only ~5% stroke cases in this dataset). Standard models fail to identify the minority class, achieving high accuracy by simply predicting "no stroke" for everyone.

### 🔑 Key Features

- ✅ **Zero Data Leakage Prevention**: Strict train/test split before any data preprocessing or SMOTE application
- ✅ **Synthetic Data Validation**: Quality verification of SMOTE-generated data using:
  - Kolmogorov-Smirnov (KS) Statistical Test
  - PCA & UMAP geometric projections for visual validation
- ✅ **Medical-Grade Evaluation**: Models assessed using:
  - F1 Score (balances precision and recall)
  - Cohen's Kappa (inter-rater agreement metric)
  - ROC-AUC (distinguishing stroke vs. non-stroke)
  - Confusion Matrix (for clinical sensitivity/specificity)
- ✅ **Multiple Algorithm Comparison**: Decision Trees, Random Forest, and Neural Networks
- ✅ **Comprehensive Documentation**: Full pipeline explanation with code comments

### 📊 Dataset Information

- **Total Samples**: Healthcare records with demographic and clinical features
- **Original Class Distribution**: ~95% negative (no stroke), ~5% positive (stroke)
- **Problem Type**: Binary classification with severe imbalance
- **Solution Approach**: SMOTE upsampling of minority class to 1:1 ratio

### 🛠️ Technologies Used

```
Python 3.8+
├── Data Processing: pandas, numpy
├── Machine Learning: scikit-learn
│   ├── Algorithms: Decision Tree, Random Forest, Neural Network
│   ├── SMOTE: imbalanced-learn
│   └── Metrics: Classification reports, confusion matrices
├── Visualization: matplotlib, seaborn
├── Dimensionality Reduction: PCA, UMAP
└── Statistical Testing: scipy (Kolmogorov-Smirnov)
```

### 📥 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/EfekaanBengi/Stroke-Prediction-SMOTE.git
   cd Stroke-Prediction-SMOTE
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install manually:
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn scipy umap-learn jupyter
   ```

3. **Run the notebook:**
   ```bash
   jupyter notebook Stroke_Prediction_SMOTE.ipynb
   ```

### 🚀 Quick Start

The project is organized as a Jupyter notebook with these main sections:

1. **Data Loading & Exploration**: Understanding dataset structure and class distribution
2. **EDA (Exploratory Data Analysis)**: Statistical analysis and visualizations
3. **Data Preprocessing**: Feature scaling, handling missing values
4. **Train/Test Split**: Separation before any balancing
5. **SMOTE Application**: Synthetic data generation on training set only
6. **Model Training**: Multiple algorithms tested
7. **Evaluation & Comparison**: Comprehensive metrics and visualizations
8. **Synthetic Data Validation**: KS test, PCA/UMAP visualizations

### 📈 Results Summary

#### Model Performance Comparison

| Model | Version | Accuracy | Recall (Stroke) | F1 Score | ROC-AUC |
|-------|---------|----------|-----------------|----------|---------|
| Decision Tree | Original | 0.9080 | 0.0784 | 0.0300 | 0.5152 |
| Decision Tree | SMOTE | 0.8894 | 0.1630 | 0.1080 | 0.5367 |
| **Random Forest** | **Original** | **0.9462** | **0.0000** | **-0.0090** | **0.5421** |
| **Random Forest** | **SMOTE** | **0.8816** | **0.1879** | **0.1315** | **0.6385** ⭐ |
| Neural Network | Original | 0.9511 | 0.0000 | 0.0000 | 0.5000 |
| Neural Network | SMOTE | 0.7613 | 0.2375 | 0.1689 | 0.6012 |

**🏆 Champion Model**: Random Forest with SMOTE (Balanced) - Best trade-off between precision and recall

#### Key Insights

- **Before SMOTE**: Models achieve ~95% accuracy but fail to detect any stroke cases (F1 ≈ 0)
- **After SMOTE**: Accuracy decreases to ~85-88%, but recall jumps significantly - models now actually detect stroke patients
- **Clinical Implication**: Catching one stroke patient is worth tolerating a few false alarms in medical diagnostics
- **Minority Class Detection**: SMOTE increased stroke case detection from 0% to ~18-23%

### 📊 Visualizations

The project includes several visualizations:

1. **PCA & UMAP Distribution**
   - Shows synthetic data (orange) integrating naturally with real stroke cases (blue)
   - Validates that SMOTE doesn't create outlier artifacts

2. **Confusion Matrix**
   - Displays True Positives, False Positives, True Negatives, False Negatives
   - Highlights importance of minimizing False Negatives (missed stroke cases)

3. **ROC Curve**
   - Demonstrates model's ability to distinguish stroke vs. non-stroke patients
   - AUC metric shows discrimination capability

4. **Model Comparison Charts**
   - Side-by-side performance metrics before/after SMOTE
   - Visual proof of effectiveness

### 💡 Learning Outcomes

This project teaches:

- How to identify and handle class imbalance in machine learning
- Why accuracy alone is misleading for imbalanced datasets
- Proper data leakage prevention techniques
- SMOTE mechanics and validation
- Clinical/domain-aware model evaluation
- Importance of multiple evaluation metrics

### 📚 References & Concepts

- **SMOTE Paper**: Chawla, N. V., et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique"
- **Imbalanced Learning**: https://imbalanced-learn.org/
- **Medical ML Best Practices**: Importance of recall/sensitivity in diagnostic tasks
- **Evaluation Metrics**: F1 Score, Cohen's Kappa, ROC-AUC for imbalanced data

### 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs or suggest improvements via Issues
- Submit pull requests with enhancements
- Improve documentation or visualizations
- Add additional algorithms or preprocessing techniques

### 📄 License

This project is open source and available under the MIT License.

### ✨ Author

**Efekaan Bengi** - [GitHub Profile](https://github.com/EfekaanBengi)

---

---

## 🇹🇷 Türkçe Versiyon

### 🎯 Proje Özeti

Tıbbi veri setlerinde **sınıf dengesizliğini** (class imbalance) SMOTE (Synthetic Minority Over-sampling Technique) kullanarak etkili bir şekilde işleyen, **inme tahminlemesi** için uçtan uca bir makine öğrenmesi boru hattı. Bu proje, yanıltıcı yüksek doğruluk (accuracy) puanları yerine, azınlık sınıfını (inme hastaları) tespit etmeye odaklanan sağlam tanısal modeller nasıl oluşturulacağını gösterir.

**Çözülen Ana Problem:** Tıbbi veri setleri genellikle ciddi sınıf dengesizliğinden muzdarip dir (bu veri setinde sadece ~%5 inme vakası). Standart modeller azınlık sınıfını tanımakta başarısız olarak, herkes için "inme yok" tahminlemesi yaparak yüksek doğruluk elde ederler.

### 🔑 Öne Çıkan Özellikler

- ✅ **Sıfır Veri Sızıntısı Önleme**: Herhangi bir veri ön işleme veya SMOTE uygulanmadan önce katı train/test ayrımı
- ✅ **Sentetik Veri Doğrulaması**: SMOTE tarafından oluşturulan verilerin kalitesi kontrol:
  - Kolmogorov-Smirnov (KS) İstatistiksel Testi
  - PCA & UMAP geometrik izdüşümleri (görsel doğrulama)
- ✅ **Tıbbi Standartında Değerlendirme**: Modeller aşağıdaki metrikleri kullanarak değerlendirilir:
  - F1 Skoru (hassasiyet ve duyarlılığı dengeler)
  - Cohen's Kappa (uyum metriği)
  - ROC-AUC (inme vs. inme yok ayrımı)
  - Karmaşıklık Matrisi (klinik duyarlılık/özgüllük)
- ✅ **Çoklu Algoritma Karşılaştırması**: Karar Ağacı, Random Forest, Yapay Sinir Ağları
- ✅ **Kapsamlı Dokümantasyon**: Kod yorumları ile tam pipeline açıklaması

### 📊 Veri Seti Bilgileri

- **Toplam Örnekler**: Demografik ve klinik özellikler içeren sağlık kayıtları
- **Orijinal Sınıf Dağılımı**: ~%95 negatif (inme yok), ~%5 pozitif (inme var)
- **Problem Türü**: Ciddi dengesizliği olan ikili sınıflandırma
- **Çözüm Yaklaşımı**: Azınlık sınıfını 1:1 oranına getirmek için SMOTE üst örnekleme

### 🛠️ Kullanılan Teknolojiler

```
Python 3.8+
├── Veri İşleme: pandas, numpy
├── Makine Öğrenmesi: scikit-learn
│   ├── Algoritmalar: Karar Ağacı, Random Forest, Yapay Sinir Ağı
│   ├── SMOTE: imbalanced-learn
│   └── Metrikler: Sınıflandırma raporları, karmaşıklık matrisleri
├── Görselleştirme: matplotlib, seaborn
├── Boyutluluk İndirgeme: PCA, UMAP
└── İstatistiksel Test: scipy (Kolmogorov-Smirnov)
```

### 📥 Kurulum

1. **Depoyu klonlayın:**
   ```bash
   git clone https://github.com/EfekaanBengi/Stroke-Prediction-SMOTE.git
   cd Stroke-Prediction-SMOTE
   ```

2. **Bağımlılıkları yükleyin:**
   ```bash
   pip install -r requirements.txt
   ```
   
   Veya manuel olarak:
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn scipy umap-learn jupyter
   ```

3. **Jupyter notebook'u çalıştırın:**
   ```bash
   jupyter notebook Stroke_Prediction_SMOTE.ipynb
   ```

### 🚀 Hızlı Başlangıç

Proje, bu ana bölümleri içeren bir Jupyter notebook olarak organize edilmiştir:

1. **Veri Yükleme ve Keşif**: Veri seti yapısı ve sınıf dağılımını anlama
2. **KED (Keşifsel Veri Analizi)**: İstatistiksel analiz ve görselleştirmeler
3. **Veri Ön İşleme**: Özellik ölçeklendirmesi, eksik değerlerin işlenmesi
4. **Train/Test Ayrımı**: Herhangi bir dengeleme öncesi ayrım
5. **SMOTE Uygulaması**: Sadece eğitim seti üzerinde sentetik veri üretimi
6. **Model Eğitimi**: Çoklu algoritmalar test edilir
7. **Değerlendirme ve Karşılaştırma**: Kapsamlı metrikler ve görselleştirmeler
8. **Sentetik Veri Doğrulaması**: KS testi, PCA/UMAP görselleştirmeleri

### 📈 Sonuçlar Özeti

#### Model Performans Karşılaştırması

| Model | Versiyon | Doğruluk | Duyarlılık (İnme) | F1 Skoru | ROC-AUC |
|-------|----------|----------|-------------------|----------|---------|
| Karar Ağacı | Orijinal | 0.9080 | 0.0784 | 0.0300 | 0.5152 |
| Karar Ağacı | SMOTE | 0.8894 | 0.1630 | 0.1080 | 0.5367 |
| **Random Forest** | **Orijinal** | **0.9462** | **0.0000** | **-0.0090** | **0.5421** |
| **Random Forest** | **SMOTE** | **0.8816** | **0.1879** | **0.1315** | **0.6385** ⭐ |
| Yapay Sinir Ağı | Orijinal | 0.9511 | 0.0000 | 0.0000 | 0.5000 |
| Yapay Sinir Ağı | SMOTE | 0.7613 | 0.2375 | 0.1689 | 0.6012 |

**🏆 Şampiyon Model**: SMOTE ile Random Forest (Dengelenmiş) - Hassasiyet ve duyarlılık arasında en iyi denge

#### Temel İçgörüler

- **SMOTE Öncesi**: Modeller ~%95 doğruluk elde ederler ancak hiçbir inme vakasını tespit edemezler (F1 ≈ 0)
- **SMOTE Sonrası**: Doğruluk ~%85-88'e düşer, ancak duyarlılık önemli ölçüde artar - modeller artık gerçekten inme hastalarını tespit ederler
- **Klinik Anlamı**: Bir inme hastası yakalamak, birkaç yanlış alarmı tolere etmeye değer
- **Azınlık Sınıfı Tespiti**: SMOTE, inme vakası tespitini %0'dan ~%18-23'e çıkardı

### 📊 Görselleştirmeler

Proje birkaç görselleştirme içerir:

1. **PCA & UMAP Dağılımı**
   - Sentetik verilerin (turuncu) gerçek inme vakalarının (mavi) uzayına nasıl doğal olarak entegre olduğunu gösterir
   - SMOTE'nin aykırı değer yaratmadığını doğrular

2. **Karmaşıklık Matrisi**
   - Doğru Pozitif, Yanlış Pozitif, Doğru Negatif, Yanlış Negatifleri gösterir
   - Yanlış Negatifleri (gözden kaçan inme) minimize etmenin önemini vurgular

3. **ROC Eğrisi**
   - Modelin inme vs. inme yok hastalarını ayırma yeteneğini gösterir
   - AUC metriği ayırım yeteneğini gösterir

4. **Model Karşılaştırma Grafikleri**
   - SMOTE öncesi/sonrası yan yana performans metrikleri
   - Etkinliğin görsel kanıtı

### 💡 Öğrenme Çıktıları

Bu proje şunları öğretir:

- Makine öğrenmesinde sınıf dengesizliğini nasıl tanımlanır ve işlenilir
- Neden sadece doğruluk dengesiz veri setleri için yanıltıcıdır
- Uygun veri sızıntısı önleme teknikleri
- SMOTE mekaniği ve doğrulanması
- Klinik/etki alanı farkındalığı ile model değerlendirmesi
- Birden fazla değerlendirme metriğinin önemi

### 📚 Referanslar ve Kavramlar

- **SMOTE Makalesi**: Chawla, N. V., vd. (2002). "SMOTE: Synthetic Minority Over-sampling Technique"
- **Dengesiz Öğrenme**: https://imbalanced-learn.org/
- **Tıbbi ML En İyi Uygulamalar**: Tanısal görevlerde duyarlılık/özgüllüğün önemi
- **Değerlendirme Metrikleri**: Dengesiz veri için F1 Skoru, Cohen's Kappa, ROC-AUC

### 🤝 Katkıda Bulunma

Katkılar açıktır! Lütfen:
- Sorunları bildirin veya Sorunlar aracılığıyla iyileştirme önerileri yapın
- Geliştirmelerle çekme istekleri gönderin
- Dokümantasyon veya görselleştirmeleri iyileştirin
- Ek algoritmalar veya ön işleme teknikleri ekleyin

### 📄 Lisans

Bu proje açık kaynak olup MIT Lisansı altında mevcuttur.

### ✨ Yazar

**Efekaan Bengi** - [GitHub Profili](https://github.com/EfekaanBengi)

---

## 📞 İletişim & Destek

Sorularınız veya önerileriniz için:
- GitHub Issues aracılığıyla iletişime geçin
- [Profil sayfanızı](https://github.com/EfekaanBengi) ziyaret edin

---

**⭐ Bu proje size yardımcı olduysa, lütfen bir yıldız vermeyi unutmayın!**
