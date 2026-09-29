# Conversion Quality Router — Proje Şartnamesi

**Belge durumu:** Uygulama öncesi kapsam kilidi  
**Teslim süresi:** 4 gün  
**Proje tipi:** Deep Learning → LLM Decision Agent → n8n Automation  
**Ana tema:** E-ticaret oturumlarında dönüşüm kalitesi tahmini ve insan denetimli aksiyon yönlendirme

---

## 1. Yönetici özeti

Conversion Quality Router, bir e-ticaret oturumunun satın alma ile sonuçlanma olasılığını kendi eğitilmiş deep learning modeliyle tahmin eder. Model çıktısı, sabit JSON şemasına uyan bir LLM karar agent'ı tarafından operasyonel bir karara çevrilir. n8n, bu kararı güven ve risk seviyesine göre loglama, bildirim veya insan onayı akışına yönlendirir.

Projenin temel iddiası şudur:

> Ham trafik hacmi tek başına değer değildir. Oturum davranışından türetilen, kalibre edilmiş dönüşüm olasılığı ve belirsizlik bilgisi daha güvenli ve ölçülebilir operasyon kararlarına dönüştürülebilir.

Sistem bir pazarlama kampanyasını otomatik olarak başlatmaz ve kullanıcıya doğrudan mesaj göndermez. Yüksek etkili veya belirsiz kararları insan incelemesine taşır. Böylece model tahmini, agent kararı ve otomasyon aksiyonu birbirinden ayrıştırılabilir ve denetlenebilir kalır.

---

## 2. Problem tanımı

### 2.1 İş problemi

E-ticaret ekipleri çok sayıda oturumu aynı şekilde değerlendirdiğinde üç sorun oluşur:

1. Yüksek trafik, yüksek dönüşüm kalitesiyle karıştırılır.
2. Belirsiz tahminler güvenilir tahminlerle aynı otomasyon akışına girer.
3. Bir model skoru, hangi operasyonel aksiyonun neden alındığını açıklamaz.

### 2.2 Teknik problem

Bir oturumun davranışsal ve teknik özelliklerinden `Revenue` sonucunu tahmin eden ikili sınıflandırma modeli kurulacaktır. Modelin görevi yalnızca olasılık üretmektir. Agent'ın görevi bu olasılığı ve izin verilen bağlamı kullanarak sınırlı bir karar vermektir. n8n'in görevi ise kararı doğrulanabilir bir dış aksiyona dönüştürmektir.

### 2.3 Araştırma soruları

- Basit bir logistic regression baseline'ına kıyasla MLP modeli dönüşüm sınıfını ne ölçüde daha iyi ayırır?
- Sınıf dengesizliği altında PR-AUC, recall ve precision nasıl değişir?
- Kalibre edilmiş olasılık ve belirsizlik bandı, otomatik karar ile insan incelemesi arasındaki sınırı güvenilir biçimde kurabilir mi?
- Yapılandırılmış agent çıktısı ve şema doğrulaması, model sonucunun otomasyonda bozulmasını önleyebilir mi?

### 2.4 Başarı tanımı

Proje başarılı sayılırsa:

- eğitim, validation ve test ayrımı yeniden üretilebilir;
- logistic regression baseline ve MLP aynı veri bölünmesinde karşılaştırılır;
- ana metrik accuracy değil PR-AUC olur;
- confusion matrix, calibration ve hata analizi raporlanır;
- API gerçek bir örnek için tahmin döndürür;
- agent her zaman sabit JSON şemasına uyar veya güvenli fallback'e düşer;
- n8n webhook'tan dış aksiyona kadar canlı çalışır;
- düşük güvenli kararlar otomatik aksiyon yerine insan incelemesine gider;
- workflow JSON'u başka bir n8n kurulumuna import edilebilir.

---

## 3. Kapsam kilidi

### 3.1 Dahil olanlar

- Tek açık veri seti
- Tek sabit veri işleme pipeline'ı
- Tek baseline: logistic regression
- Tek deep learning modeli: MLP
- Tek model artifact seti
- Tek FastAPI servisi
- Tek LLM karar agent'ı
- Tek sabit JSON çıktı şeması
- Tek n8n workflow'u
- En az bir gerçek dış servis aksiyonu
- İnsan onayı ve hata fallback'i
- Model değerlendirme notebook'u veya script'i
- Testler, README, workflow export ve demo videosu

### 3.2 Dahil olmayanlar

- Web scraping
- Gerçek zamanlı veri toplama
- Frontend veya dashboard
- Kullanıcı girişi ve yetkilendirme sistemi
- RAG veya vector database
- Çoklu agent mimarisi
- Birden fazla deep learning mimarisi
- Hyperparameter search platformu
- Otomatik yeniden eğitim
- Gerçek müşteriye otomatik kampanya gönderimi
- Kişisel veya hassas sağlık verisi
- Mobil uygulama
- Zorunlu cloud deployment

Bu liste dördüncü gün sonuna kadar dondurulmuştur. Zorunlu kabul kriterlerinden biri tamamlanmadan bonus özellik alınmaz.

---

## 4. Veri seti

### 4.1 Kaynak

**Online Shoppers Purchasing Intention Dataset**  
UCI Machine Learning Repository, dataset ID 468  
DOI: `10.24432/C5F88Q`  
Lisans: CC BY 4.0

Kaynak: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

### 4.2 Veri özeti

- 12.330 ayrı kullanıcı oturumu
- 17 tahmin değişkeni
- Hedef: `Revenue`
- 10.422 negatif sınıf
- 1.908 pozitif sınıf
- Eksik değer yok
- Sayısal ve kategorik değişkenlerin birlikte bulunduğu tabular veri

### 4.3 Önemli değişken grupları

- Sayfa sayıları: `Administrative`, `Informational`, `ProductRelated`
- Sayfa süreleri: ilgili `Duration` alanları
- Davranış sinyalleri: `BounceRates`, `ExitRates`
- Ticari bağlam: `SpecialDay`, `Month`, `VisitorType`, `Weekend`
- Teknik bağlam: `OperatingSystems`, `Browser`, `Region`, `TrafficType`
- İncelenecek yüksek riskli alan: `PageValues`

### 4.4 Leakage kontrolü

`PageValues` hedefe çok yakın veya karar anından sonra oluşabilecek bilgi taşıyabilir. Bu nedenle:

1. Özelliğin hesaplanma zamanı veri dokümantasyonundan açıklanır.
2. Ana/deployment-safe deneyde `PageValues` varsayılan olarak dışarıda bırakılır.
3. Zaman kalırsa yalnızca sensitivity analizi olarak dahil edildiği bir karşılaştırma yapılır.
4. Sonuçlarda performans ile operasyonel geçerlilik arasındaki fark açıkça yazılır.

Agent veya otomasyon katmanı model girdisine ek bilgi ekleyemez.

### 4.5 Veri bölünmesi

- Train: %70
- Validation: %15
- Test: %15
- Stratified split
- Sabit random seed
- Preprocessing yalnızca train verisine fit edilir
- Test setine model/threshold seçimi sırasında bakılmaz

Veride gerçek zaman damgası bulunmadığından zamansal holdout ana tasarım değildir. Bu sınırlılık README ve sunumda belirtilir.

---

## 5. Modelleme planı

### 5.1 Baseline

Logistic regression:

- aynı train/validation/test split;
- aynı leakage-safe özellik seti;
- class weighting;
- kategorik değişkenler için one-hot encoding;
- standardize edilmiş sayısal değişkenler;
- validation setinde threshold seçimi.

Baseline'ın amacı yalnızca bir sayı üretmek değil, MLP'nin gerçekten ek değer sağlayıp sağlamadığını göstermektir.

### 5.2 Deep learning modeli

PyTorch tabular MLP:

- giriş katmanı: preprocessing sonrası feature vektörü;
- hidden layers: örneğin `128 → 64 → 32`;
- activation: ReLU;
- dropout: 0.2–0.4 aralığında tek seçilmiş değer;
- output: tek logit;
- loss: `BCEWithLogitsLoss` ve train sınıf dağılımından hesaplanan `pos_weight`;
- optimizer: Adam;
- early stopping: validation PR-AUC veya validation loss;
- maksimum epoch sınırı;
- CPU üzerinde çalışabilir eğitim;
- random seed ve deterministik ayarlar.

Yalnızca küçük, gerekçeli ayarlar denenir. Geniş grid search yapılmaz.

### 5.3 Ana metrikler

**Primary:**

- PR-AUC

**Secondary:**

- ROC-AUC
- Precision
- Recall
- F1
- Confusion matrix
- Brier score
- Calibration curve

Accuracy yalnızca tamamlayıcı olarak raporlanabilir; ana başarı ölçüsü olamaz.

### 5.4 Threshold seçimi

Threshold test setinden seçilmez. Validation setinde aşağıdaki iş maliyeti mantığı kullanılır:

- Yanlış negatif: değerli bir oturumun kaçırılması
- Yanlış pozitif: gereksiz operasyon incelemesi
- Belirsiz tahmin: otomasyona güvenli biçimde verilemeyen örnek

Seçilen threshold ve seçim gerekçesi model metadata dosyasına kaydedilir.

### 5.5 Olasılık ve belirsizlik

Ham sigmoid çıktısı doğrudan “güven” olarak adlandırılmaz.

- Olasılık validation verisiyle kalibre edilir.
- `purchase_probability` tahmin edilen sonuç olasılığıdır.
- `decision_margin`, olasılığın karar eşiğine uzaklığıdır.
- `uncertain`, kalibrasyon ve eşik çevresindeki önceden tanımlı banda göre hesaplanır.
- Belirsizlik kuralı kodda deterministiktir; LLM tarafından belirlenmez.

### 5.6 Hata analizi

En az şu kırılımlar incelenir:

- yeni ve geri dönen ziyaretçiler;
- hafta içi ve hafta sonu;
- aylar;
- traffic type;
- false positive örnekler;
- false negative örnekler;
- düşük ve yüksek olasılık dilimleri;
- calibration dilimleri.

Model başarısızlıkları saklanmaz. Sunumda en az iki somut hata örneği gösterilir.

---

## 6. Model çıktı sözleşmesi

Örnek model çıktısı:

```json
{
  "request_id": "req_20260928_001",
  "prediction": {
    "purchase_probability": 0.73,
    "decision_threshold": 0.58,
    "predicted_class": "likely_to_convert",
    "decision_margin": 0.15,
    "uncertain": false
  },
  "signals": [
    "returning_visitor",
    "long_product_related_duration",
    "low_exit_rate"
  ],
  "model": {
    "name": "conversion_mlp",
    "version": "1.0.0",
    "feature_contract_version": "1.0.0"
  }
}
```

`signals` alanı yalnızca izin verilen, deterministik olarak hesaplanan reason code'lardan oluşur. Serbest metin feature açıklaması model katmanında üretilmez.

---

## 7. LLM karar agent'ı

### 7.1 Agent'ın görevi

Agent model sonucunu yeniden tahmin etmez. Aşağıdaki sınırlı kararlardan birini seçer:

- `LOG_ONLY`
- `PRIORITY_REVIEW`
- `HUMAN_REVIEW`
- `SYSTEM_FALLBACK`

### 7.2 Agent'ın yapamayacakları

- Model olasılığını değiştiremez.
- Yeni feature veya kullanıcı bilgisi uyduramaz.
- İzin verilen listenin dışında aksiyon öneremez.
- Belirsiz tahmini yüksek güvenli gibi sunamaz.
- Kullanıcıya otomatik pazarlama mesajı gönderemez.
- JSON dışında çıktı üretemez.

### 7.3 Sabit agent çıktı şeması

```json
{
  "request_id": "req_20260928_001",
  "decision": "HUMAN_REVIEW",
  "priority": "MEDIUM",
  "allowed_action": "ADD_TO_REVIEW_QUEUE",
  "requires_human_approval": true,
  "reason_codes": [
    "MODEL_UNCERTAIN",
    "NEAR_DECISION_THRESHOLD"
  ],
  "explanation": "Tahmin karar eşiğine yakın olduğu için otomatik önceliklendirme yapılmadı.",
  "agent_version": "1.0.0",
  "schema_version": "1.0.0"
}
```

### 7.4 Hallucination koruması

- JSON Schema/Pydantic doğrulaması
- Enum kısıtlamaları
- Maksimum açıklama uzunluğu
- Model çıktısıyla çapraz alan kontrolü
- İzin verilen reason code listesi
- Bir kez kontrollü retry
- İkinci hatada deterministic fallback
- Her ham cevap yerine yalnızca güvenli, normalize edilmiş çıktı loglanır

### 7.5 Deterministik fallback

Agent çağrısı başarısız veya şema dışıysa:

```json
{
  "decision": "SYSTEM_FALLBACK",
  "priority": "HIGH",
  "allowed_action": "ADD_TO_REVIEW_QUEUE",
  "requires_human_approval": true,
  "reason_codes": ["AGENT_OUTPUT_INVALID"],
  "explanation": "Agent çıktısı doğrulanamadı; kayıt insan incelemesine aktarıldı."
}
```

Bu fallback bir LLM çağrısı değildir.

---

## 8. API tasarımı

### 8.1 Endpoint'ler

- `GET /health`
- `GET /ready`
- `POST /predict`
- `POST /decide`
- `POST /route`

`/route`, demo için model ve agent zincirini tek çağrıda çalıştırabilir. Ayrı endpoint'ler test edilebilirliği korur.

### 8.2 Girdi doğrulaması

- Pydantic modelleri
- Kategori whitelist'leri
- Sayısal aralık kontrolleri
- Bilinmeyen alanları reddetme
- Request ID üretimi
- Secret veya kişisel veri loglamama

### 8.3 Hata sözleşmesi

```json
{
  "request_id": "req_20260928_001",
  "error": {
    "code": "MODEL_UNAVAILABLE",
    "message": "Prediction service is temporarily unavailable.",
    "retryable": true
  }
}
```

HTTP durum kodları ve hata kodları README'de açıklanır.

---

## 9. n8n workflow tasarımı

### 9.1 Ana akış

1. Webhook test girdisini alır.
2. Input validation uygulanır.
3. FastAPI `/route` veya `/predict` çağrılır.
4. Agent sonucu JSON olarak doğrulanır.
5. Switch node `decision` alanına göre dallanır.
6. Dış aksiyon uygulanır.
7. Sonuç loglanır.
8. Webhook response döndürülür.

### 9.2 Dallar

| Karar | Aksiyon | İnsan onayı |
|---|---|---:|
| `LOG_ONLY` | Kayıt tablosuna ekle | Hayır |
| `PRIORITY_REVIEW` | Öncelikli kayıt + ekip bildirimi | Tercihen |
| `HUMAN_REVIEW` | Onay kuyruğu + bildirim | Evet |
| `SYSTEM_FALLBACK` | Hata kaydı + yüksek öncelikli bildirim | Evet |

### 9.3 Dış servis

Birinci gün sonunda aşağıdakilerden erişilebilir olanı kilitlenir:

1. Google Sheets + e-posta
2. Google Sheets + Slack/Discord webhook
3. Başka bir gerçek kayıt servisi + ekip bildirimi

En az bir gerçek dış servis zorunludur. Sadece n8n ekran görüntüsü veya manuel node çalıştırma kabul edilmez.

### 9.4 Hata yönetimi

- API timeout
- API 4xx/5xx
- Geçersiz agent JSON
- Dış servis credential hatası
- Duplicate `request_id`
- Retry sınırı
- Error workflow veya fallback dalı

Workflow export'u `automation/conversion_quality_router.json` altında tutulur ve temiz bir n8n kurulumunda import testi yapılır.

---

## 10. Teknik yığın

- Python 3.11
- PyTorch
- pandas / NumPy
- scikit-learn
- FastAPI / Uvicorn
- Pydantic
- httpx tabanlı provider-neutral LLM client
- n8n
- pytest
- Ruff
- Jupyter veya yeniden üretilebilir Python evaluation script'i

Secret'lar yalnızca environment variable olarak kullanılır. `.env` hiçbir zaman repoya eklenmez; yalnızca `.env.example` bulunur.

---

## 11. Önerilen repo yapısı

```text
conversion-quality-router/
├── README.md
├── LICENSE
├── pyproject.toml
├── .env.example
├── data/
│   ├── README.md
│   └── sample/
├── docs/
│   ├── architecture.md
│   ├── model-card.md
│   ├── agent-card.md
│   ├── evaluation.md
│   └── demo-script.md
├── notebooks/
│   └── 01_end_to_end_analysis.ipynb
├── src/conversion_router/
│   ├── config.py
│   ├── schemas.py
│   ├── data/
│   │   ├── load.py
│   │   ├── validate.py
│   │   └── preprocess.py
│   ├── modeling/
│   │   ├── baseline.py
│   │   ├── network.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── calibrate.py
│   │   └── inference.py
│   ├── agent/
│   │   ├── prompt_v1.md
│   │   ├── client.py
│   │   ├── validator.py
│   │   └── fallback.py
│   └── api/
│       ├── app.py
│       └── routes.py
├── artifacts/
│   ├── README.md
│   └── metadata/
├── automation/
│   ├── README.md
│   └── conversion_quality_router.json
├── scripts/
│   ├── download_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── smoke_test.py
└── tests/
    ├── unit/
    ├── integration/
    └── fixtures/
```

Notebook kanıt ve anlatım içindir; model mantığını yeniden yazmaz, `src/` modüllerini ve dondurulmuş artifact/prediction dosyalarını kullanır. Üretim mantığının tek kopyası `src/` altında bulunur. Tek, baştan sona çalıştırılmış ve çıktıları görünür `notebooks/01_end_to_end_analysis.ipynb` dosyası zorunlu teslim parçasıdır (bkz. `.project-control/DECISIONS.md` D08 ve T305).

---

## 12. Dört günlük uygulama programı

### Gün 1 — Kapsam, veri, EDA ve baseline

**Sabah**

- Repo scaffold
- Environment ve bağımlılıkların doğrulanması
- Veri indirme ve checksum
- Data dictionary
- Target dağılımı
- Leakage ve veri kalitesi incelemesi

**Öğleden sonra**

- Train/validation/test split
- Preprocessing pipeline
- Logistic regression baseline
- İlk PR-AUC, ROC-AUC ve confusion matrix
- Dış servis ve agent endpoint erişiminin doğrulanması

**Gün 1 kapısı**

- Veri tek komutla hazırlanıyor.
- Split'ler sabit ve kaydedilmiş.
- Baseline sonucu tekrar üretilebiliyor.
- Leakage kararı yazılı.
- Dış servis seçilmiş ve credential yolu doğrulanmış.

### Gün 2 — MLP, değerlendirme ve FastAPI

**Sabah**

- PyTorch Dataset/DataLoader
- MLP eğitimi
- Early stopping
- Model artifact ve metadata

**Öğleden sonra**

- Baseline karşılaştırması
- Calibration
- Threshold ve uncertainty band
- Error analysis
- `/health`, `/ready`, `/predict`

**Gün 2 kapısı**

- Model test setinde bir kez değerlendirilmiş.
- Bütün ana metrikler ve confusion matrix mevcut.
- Model artifact API tarafından yüklenebiliyor.
- Örnek input için deterministik prediction JSON dönüyor.

### Gün 3 — Agent ve n8n entegrasyonu

**Sabah**

- Agent prompt v1
- JSON Schema/Pydantic model
- Cross-field validation
- Retry ve deterministic fallback
- Agent unit testleri

**Öğleden sonra**

- n8n webhook
- API çağrısı
- Switch dalları
- Dış servis aksiyonu
- Human approval
- Error path
- Workflow export

**Gün 3 kapısı — özellik dondurma**

- Webhook'tan dış aksiyona kadar uçtan uca akış çalışıyor.
- Dört karar dalı test edilmiş.
- Invalid agent output güvenli fallback'e düşüyor.
- Workflow JSON export edilmiş.
- Bundan sonra yeni özellik eklenmiyor.

### Gün 4 — QA, kanıt, dokümantasyon ve demo

**Sabah**

- Unit ve integration testleri
- En az beş uçtan uca demo fixture'ı
- Temiz ortam kurulumu/smoke test
- Workflow import testi
- Secret ve attribution audit

**Öğleden sonra**

- README
- Model card ve agent card
- Evaluation report
- Mimari diyagram
- 2–3 dakikalık demo videosu
- 20 dakikalık sunum provası
- Son hata düzeltmeleri

**Gün 4 kapısı**

- Teslim checklist'i eksiksiz.
- Repo sıfırdan kurulabiliyor.
- Testler geçiyor.
- Demo kayıtlı.
- Sunumda model hatası, agent hatası ve şirket ortamında ilk değişiklik soruları cevaplanabiliyor.

---

## 13. Faz kapıları

| Faz | Tamamlanma kanıtı | Sonraki faza geçiş şartı |
|---|---|---|
| Scope | Dondurulmuş şartname | Non-goal listesi kabul edilmiş |
| Data | Schema, kalite raporu, split | Leakage kararı yazılı |
| Baseline | Metrikler ve artifact | Tek komutla tekrar üretim |
| DL model | Test metrikleri, calibration, errors | Model metadata eksiksiz |
| API | Contract testleri | `/predict` smoke test geçiyor |
| Agent | Schema ve adversarial testler | Invalid output fallback çalışıyor |
| n8n | Export + canlı dış aksiyon | Tüm Switch dalları çalışıyor |
| QA | Test raporu + clean install | Kritik hata yok |
| Delivery | README, video, sunum | Teslim paketi açılabiliyor |

Bir faz yalnızca dosya oluşturulduğu için tamamlanmış sayılmaz. Kanıt üretilmiş ve doğrulanmış olmalıdır.

---

## 14. Test stratejisi

### 14.1 Unit testleri

- Feature schema validation
- Preprocessing output shape
- Bilinmeyen kategori davranışı
- Threshold kararı
- Uncertainty band
- Agent JSON schema
- Cross-field validation
- Deterministic fallback
- Secret redaction

### 14.2 Model testleri

- Eğitim smoke test
- Sabit seed ile benzer sonuç
- Model artifact load
- Prediction range `[0, 1]`
- Batch ve single prediction uyumu
- Test setinin training sırasında kullanılmadığının kontrolü

### 14.3 API testleri

- Health/ready
- Geçerli istek
- Eksik alan
- Bilinmeyen alan
- Aralık dışı değer
- Model unavailable
- Agent unavailable
- Invalid agent response

### 14.4 n8n test matrisi

| Senaryo | Beklenen karar | Beklenen aksiyon |
|---|---|---|
| Düşük olasılık, yüksek güven | `LOG_ONLY` | Sadece kayıt |
| Yüksek olasılık, yüksek güven | `PRIORITY_REVIEW` | Kayıt + bildirim |
| Eşik çevresi | `HUMAN_REVIEW` | Onay kuyruğu |
| Agent invalid JSON | `SYSTEM_FALLBACK` | Hata + insan incelemesi |
| API timeout | Hata dalı | Retry sınırı + bildirim |

### 14.5 Kabul testleri

- Yeni ortamda kurulum
- Eğitim veya sağlanan artifact ile inference
- API başlatma
- n8n workflow import
- Beş fixture ile uçtan uca çalışma
- Dış serviste kaydın/bildirimin görülmesi

---

## 15. Değerlendirme kriterleriyle eşleme

| Kriter | Projedeki kanıt |
|---|---|
| Problem ve veri | Açık problem, lisanslı veri, data dictionary, leakage analizi |
| DL modeli | Kendi eğitilmiş PyTorch MLP, train/val/test, early stopping |
| Baseline | Aynı split'te logistic regression karşılaştırması |
| Doğru metrik | PR-AUC ana metrik, calibration ve confusion matrix |
| Hata analizi | FP/FN örnekleri ve segment kırılımları |
| Agent | Sabit schema, prompt versioning, belirsizlik ve fallback |
| Hallucination guard | Enum, cross-field validation, retry sınırı |
| n8n | Webhook, Switch, dış servis, human approval, error path |
| Entegrasyon | Canlı uçtan uca demo |
| Repo kalitesi | README, modüler kaynak kod, testler, cards |
| Teslim | Notebook/evaluation, prompt, workflow JSON, demo video |
| Sunum | Katman bazlı anlatım ve Q&A hazırlığı |

Hedef seviye, yalnızca sabit rapor üreten “basic” çözüm değil; confidence routing ve human approval içeren “medium” çözümdür.

---

## 16. Risk kaydı

| Risk | Olasılık | Etki | Erken sinyal | Önlem |
|---|---:|---:|---|---|
| MLP baseline'ı geçmez | Orta | Orta | Gün 2 ilk deney | Sonucu dürüst raporla; calibration ve error analysis değerini göster |
| `PageValues` leakage | Yüksek | Yüksek | Aşırı yüksek metrik | Ana modelden çıkar; sensitivity olarak raporla |
| LLM çıktısı schema dışı | Orta | Yüksek | Parse hatası | Strict schema, tek retry, deterministic fallback |
| API credential yok | Orta | Yüksek | Gün 1 bağlantı testi başarısız | İlk gün provider-neutral endpoint'i doğrula |
| n8n dış servis OAuth gecikmesi | Orta | Yüksek | Gün 1 credential testi başarısız | Daha basit webhook/SMTP seçeneğine geç |
| PyTorch kurulumu sorunlu | Düşük-Orta | Yüksek | İlk environment smoke test | Sabit Python sürümü; CPU wheel; erken doğrulama |
| Scope creep | Yüksek | Yüksek | Yeni özellik listesi büyür | Gün 3 feature freeze; non-goal listesi |
| Demo sırasında servis hatası | Orta | Yüksek | Smoke test kararsız | Kayıtlı video + yerel fixture + health checks |
| Secret repo içine girer | Düşük | Kritik | Git diff/audit | `.env` ignore, secret scan, örnek değerler |

---

## 17. Teslim paketi

Zorunlu:

1. GitHub repository
2. Kurulum ve çalışma adımları içeren README
3. Veri ve lisans açıklaması
4. Tek, baştan sona çalıştırılmış ve çıktıları görünür EDA/model notebook'u (`notebooks/01_end_to_end_analysis.ipynb`; D08)
5. Baseline ve MLP karşılaştırması
6. Confusion matrix ve error analysis
7. Model artifact metadata
8. Agent prompt v1
9. JSON schema
10. n8n workflow JSON
11. Testler
12. 2–3 dakikalık demo videosu
13. Sunum

Opsiyonel, ancak temel sistem tamamen bittiyse:

- Container
- Cloud deployment
- İkinci dış servis
- Daha gelişmiş model açıklanabilirliği

---

## 18. Demo senaryosu

2–3 dakikalık videoda:

1. Mimariyi 15–20 saniyede göster.
2. Yüksek güvenli negatif fixture gönder; `LOG_ONLY` dalını göster.
3. Yüksek güvenli pozitif fixture gönder; dış bildirim/kaydı göster.
4. Belirsiz fixture gönder; human approval dalını göster.
5. Invalid agent cevabını fixture ile simüle et; fallback'i göster.
6. Model metrikleri ve confusion matrix'i kısa göster.
7. Workflow JSON ve test sonucuyla bitir.

Demo canlı çalışmalı; video yalnızca teslim kanıtıdır.

---

## 19. Sunum planı

### İlk 3 dakika — Problem ve veri

- Trafik ile dönüşüm kalitesi farkı
- Veri kaynağı, lisans, sınıf dağılımı
- Leakage riski

### Sonraki 5 dakika — Model

- Split ve preprocessing
- Logistic regression baseline
- MLP mimarisi
- PR-AUC, confusion matrix, calibration
- İki hata örneği

### Sonraki 3 dakika — Agent

- Girdi/çıktı sözleşmesi
- Agent'ın karar sınırları
- Schema validation ve fallback

### Sonraki 4 dakika — Canlı demo

- Normal karar
- İnsan onayı
- Hata dalı

### Son 5 dakika — Q&A

- Model nerede başarısız olur?
- Agent yanlış karar verirse ne olur?
- Şirkette kullanılacaksa ilk hangi değişiklik yapılır?
- Veri setinin gerçek kullanım sınırlılığı nedir?

---

## 20. Definition of Done

Proje yalnızca aşağıdakilerin tamamı sağlandığında bitmiştir:

- [ ] Tek komutla veri hazırlanabiliyor.
- [ ] Baseline yeniden üretilebiliyor.
- [ ] MLP yeniden üretilebiliyor.
- [ ] Test seti yalnızca final değerlendirmede kullanıldı.
- [ ] Ana ve ikincil metrikler raporlandı.
- [ ] Confusion matrix ve calibration mevcut.
- [ ] En az iki model hatası analiz edildi.
- [ ] Model artifact ve metadata mevcut.
- [ ] API contract testleri geçiyor.
- [ ] Agent çıktısı sabit schema'ya uyuyor.
- [ ] Invalid agent output fallback'e gidiyor.
- [ ] n8n dört karar/hata dalını çalıştırıyor.
- [ ] En az bir gerçek dış servis aksiyonu görüldü.
- [ ] Workflow export edilip yeniden import edildi.
- [ ] Secret scan temiz.
- [ ] README temiz ortam adımlarını içeriyor.
- [ ] Demo videosu hazır.
- [ ] Sunum provası tamamlandı.

---

## 21. İlk oturumda verilmesi gereken kararlar

Uygulama başlamadan yalnızca şu kararlar kilitlenir:

1. Kullanılacak dış servis: Sheets + e-posta mı, webhook tabanlı bildirim mi?
2. Kullanılacak LLM endpoint'i ve erişim yönteminin çalıştığı doğrulandı mı?
3. n8n yerel mi, mevcut bir instance üzerinde mi çalışacak?
4. Repo dili tamamen İngilizce mi olacak, dokümantasyon iki dilli mi olacak?

Bu dört karar mimariyi değiştirmez. Cevap verilmezse en düşük operasyonel riskli seçenek seçilir ve karar günlüğüne yazılır.
