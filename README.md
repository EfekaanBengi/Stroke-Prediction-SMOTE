# 🧠 Stroke Prediction & Imbalanced Data Handling via SMOTE

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange)
![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-SMOTE-success)
![Data Science](https://img.shields.io/badge/Data_Science-Machine_Learning-red)

> Yüksek oranda dengesiz tıbbi verileri (class imbalance) SMOTE ile işleyen, veri sızıntısını (data leakage) önleyen uçtan uca bir makine öğrenmesi (Machine Learning) boru hattı.

## 🚀 Proje Özeti
Bu proje, tıbbi veri setlerinde sıklıkla karşılaşılan "sınıf dengesizliği" problemini çözmek üzerine odaklanmıştır. Sadece %5'i inme (stroke) geçirmiş hastalardan oluşan bir veri setinde, makine öğrenmesi modellerinin sadece "sağlıklı" sınıfını ezberlemesi engellenmiştir. Klasik *Doğruluk (Accuracy)* metriğinin yarattığı illüzyon kırılarak, azınlık sınıfını (inme vakalarını) tespit etmeye odaklanan sağlam (robust) bir analitik model kurulmuştur.

## 🔑 Öne Çıkan Özellikler

- **Sıfır Veri Sızıntısı (Zero Data Leakage):** Veri ön işleme ve SMOTE işlemleri, modelin test verisinden kopya çekmesini önlemek adına katı bir şekilde ayrıştırılmış (Train/Test Split) ve sadece eğitim seti üzerinde uygulanmıştır.
- **Sentetik Veri Doğrulaması (Validation):** SMOTE ile üretilen sentetik verilerin kalitesi **Kolmogorov-Smirnov (KS) Testi**, **PCA** ve **UMAP** geometrik izdüşümleriyle istatistiksel ve görsel olarak kanıtlanmıştır.
- **Tıbbi Hassasiyette Değerlendirme:** Modeller yanıltıcı Accuracy değerleriyle değil; sağlık projelerinde kritik olan **F1 Score**, **Cohen's Kappa**, **ROC-AUC** ve **Confusion Matrix** ile değerlendirilmiştir.

---

## 📊 Örnek Çıktılar ve Görselleştirmeler

### 1. Sınıf Dengesizliğinin Çözümü (PCA & UMAP Dağılımı)
SMOTE ile üretilen sentetik verilerin (turuncu), gerçek inme vakalarının (mavi) uzayına nasıl başarılı bir şekilde ve doğal yollarla entegre olduğunu gösteren 2 boyutlu izdüşüm testi.

![PCA ve UMAP Görseli](gorseller/pca_umap_plot.png)
*(Not: Sentetik veriler gerçek hastaların çevresinde kümelenerek başarılı bir veri artırımı sağlamıştır.)*

### 2. Şampiyon Model: Karmaşıklık Matrisi (Confusion Matrix)
En iyi performansı gösteren **Random Forest (SMOTE'lu)** modelinin test seti üzerindeki tahmin başarıları. Sağlık verilerinde asıl odaklandığımız kısım "False Negative" (Gözden kaçan hastalar) oranını minimize etmektir.

![Confusion Matrix](gorseller/confusion_matrix.png)

### 3. ROC Eğrisi ve AUC Skoru
Modelin sağlıklı ve inme riskli hastaları birbirinden ayırma yeteneğinin geometrik kanıtı.

![ROC Curve](gorseller/roc_curve.png)

---

## 🏆 Model Performans Karşılaştırması

SMOTE öncesi ve sonrası farklı algoritmaların metrik kıyaslaması:

| Model | Veri Seti | Accuracy | F1 Score | Kappa Skoru | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Karar Ağacı (Decision Tree)** | Orijinal (SMOTE Yok) | 0.9080 | 0.0784 | 0.0300 | 0.5152 |
| **Karar Ağacı (Decision Tree)** | Dengelenmiş (SMOTE Var) | 0.8894 | 0.1630 | 0.1080 | 0.5367 |
| **Random Forest** | Orijinal (SMOTE Yok) | 0.9462 | 0.0000 | -0.0090 | 0.5421 |
| **Random Forest** | **Dengelenmiş (SMOTE Var)** | **0.8816** | **0.1879** | **0.1315** | **0.6385** |
| **Yapay Sinir Ağı (NN)** | Orijinal (SMOTE Yok) | 0.9511 | 0.0000 | 0.0000 | 0.5000 |
| **Yapay Sinir Ağı (NN)** | Dengelenmiş (SMOTE Var) | 0.7613 | 0.2375 | 0.1689 | 0.6012 |

*(Tablo Yorumu: Orijinal veride %95 Accuracy alan modellerin aslında hiçbir inme vakasını bulamadığı (F1=0), SMOTE sonrası modellerin ise risk alarak gerçek teşhisler yapmaya başladığı görülmektedir.)*

---
