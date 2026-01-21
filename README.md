# Prepoznavanje uha korištenjem dubokog učenja

Ovaj repozitorij sadrži implementaciju sustava za prepoznavanje uha korištenjem konvolucijske neuronske mreže (CNN) i prijenosa učenja. Projekt je izrađen u sklopu završnog rada te obuhvaća cjelokupan postupak obrade podataka, treniranja modela i evaluacije performansi u identifikacijskom i verifikacijskom scenariju.

## Korištene tehnologije

- Python  
- PyTorch  
- torchvision  
- NumPy  
- PIL (Python Imaging Library)

## Skup podataka

Za treniranje i evaluaciju korištena je javno dostupna baza slika uha preuzeta s platforme Kaggle (Kaggle Ear Dataset).
Zbog veličine i licencnih ograničenja, slike uha nisu uključene u ovaj repozitorij.

Bazu je potrebno preuzeti na ovome linku: https://www.kaggle.com/datasets/ushasukhanyas/ear-dataset i smjestiti u direktorij Ear_dataset/, u skladu sa strukturom očekivanom u skriptama.

## Treniranje modela

Model se trenira pokretanjem sljedeće skripte:

    python train_cnn.py


Tijekom treniranja koristi se unaprijed trenirana CNN arhitektura ResNet18 uz prijenos učenja.
Nakon treniranja, model se sprema u datoteku ear_cnn.pth.

Datoteka s istreniranim modelom nije uključena u repozitorij, jer se može reproducirati ponovnim treniranjem.

## Evaluacija modela

Evaluacija performansi provodi se pokretanjem skripte:

    python evaluate_metrics.py


Evaluacija obuhvaća:

- Identifikacijski scenarij (1:N) – točnost klasifikacije

- Verifikacijski scenarij (1:1) – FAR, FRR i EER

- Rezultati evaluacije ispisuju se u konzoli.

## Napomena

Repozitorij sadrži sav izvorni kod potreban za reprodukciju rada.
Podatkovna baza i istrenirani model izostavljeni su namjerno kako bi se osigurala ponovljivost postupka i izbjegla distribucija velikih datoteka.

## Konačna struktura projekta

```text
data/
└── splits/
    ├── train.csv
    └── test.csv

Ear_dataset/                # slike uha (nije uključeno u repozitorij)

scripts/
├── dataset_preview.py
├── prepare_split.py
├── train_cnn.py
└── evaluate_metrics.py

ear_cnn.pth                 # istrenirani model (nije uključen)

README.md
