# Chatbot Knowledge - Qwen 3B

Application Streamlit de question-reponse sur des donnees SDV EV/PHEV (fichier XLSB), avec:

- recherche hybride (BM25 + embeddings)
- stockage vectoriel ChromaDB
- generation locale via modele GGUF (Qwen2.5-3b-instruct-q8)

## 1. Prerequis

- Windows (teste)
- Python 3.11
- Modeles locaux presents dans le dossier `modeles/`
- Fichier de donnees: `Extraction_Synthese_Modelisation_SDV5.xlsb` 

## 2. Installation

Dans le dossier du projet:

```powershell
pip install -r requirements.txt
```

## 3. Lancement

Option 1:

```powershell
streamlit run app_streamlit.py
```

Option 2 (batch):

```powershell
.\start_app.bat
```

## 4. Structure utile du projet

- `app_streamlit.py`: logique principale (chargement, retrieval, prompt, fallback)
- `modeles/`: embedder + modele LLM GGUF
- `sdv_chromadb/`: base Chroma persistante
- `sdv_bm25_cpu.pkl`: cache BM25

## 5. Parametres/Comportement

Le pipeline suit globalement ces etapes:

1. contextualisation de la question (gestion des suivis)
2. retrieval hybride (BM25 + dense)
3. filtrage par score
4. generation LLM
5. fallback deterministic si reponse fragile (zone non specifiee, stats manquantes, reponse trop courte)

## 6. Test manuel recommande (conversation)

Objectif: verifier le suivi conversationnel (zone/periode/comparaison) et observer les limites de pertinence.

### Etape A - question initiale

Question:

```text
Durée cumulée parking recharge rapide Europe PHEV 3 ans
```

Reponse attendue (exemple):

```text
Critère: Durée cumulée en parking branché avec recharge en rapide [h]
Zone: Europe
Periode: 3 Years or 60 000 km
Loi: Log normale
Min: 0.51
Max: 19.2
Moy: 3.77
```

Sources attendues (ordre/score proches):

```text
score=0.945 | Durée cumulée en parking branché avec recharge en rapide [h] | Europe | 3 Years or 60 000 km
score=0.945 | Durée cumulée en parking branché avec recharge en rapide [h] | Europe | 3 Years or 60 000 km
score=0.944 | Durée cumulée en parking branché avec recharge en rapide [h] | Europe | 3 Years or 60 000 km
```

### Etape B - suivi de zone

Question:

```text
et en Chine ?
```

Comportement observe (acceptable si pas de donnee suffisamment proche):

```text
Pertinence faible (score=0.62)
Critère le plus proche : → Durée [s] passée dans SDV aggravante 3 avant d'entrer dans zone critique Rumble 3/ Time spent (s) in
Reformulez avec des mots plus précis.
```

### Etape C - suivi de periode

Question:

```text
et pour 7 ans ?
```

Reponse attendue (exemple):

```text
Critère: Durée cumulée en parking branché avec recharge en rapide [h]
Zone: Europe
Periode: 7 Years or 120 000 km
Loi: Log normale
Min: 0.51
Max: 19.2
Moy: 3.77
```

Sources attendues (scores proches):

```text
score=0.953 | Durée cumulée en parking branché avec recharge en rapide [h] | Europe | 3 Years or 60 000 km
score=0.953 | Durée cumulée en parking branché avec recharge en rapide [h] | Europe | 3 Years or 60 000 km
score=0.950 | Durée cumulée en parking branché avec recharge en rapide [h] | Europe | 3 Years or 60 000 km
```

### Etape D - comparaison

Question:

```text
compare avec recharge normale
```

Reponse attendue (exemple):

```text
Critère: Durée cumulée en parking branché avec recharge en normal [h]
Zone: Europe
Periode: 7 years or 150 000 km
Loi: Log normale
Min: 773.0
Max: 7870.0
Moy: 3970.0
```

Trace contextuelle attendue:

```text
Question contextualisee: et pour 7 ans ? | Suivi: compare avec recharge normale
```

Sources attendues (scores proches):

```text
score=0.921 | Durée cumulée en parking branché avec recharge en normal [h] | Europe | 7 years or 150 000 km
score=0.921 | Durée cumulée en parking branché avec recharge en normal [h] | Europe | 7 years or 150 000 km
score=0.921 | Durée cumulée en parking branché avec recharge en normal [h] | Europe | 7 years or 150 000 km
```

## 7. Criteres de validation rapide

- Le chatbot retrouve bien le critere cible sur la question initiale.
- Le suivi conversationnel conserve le contexte utile entre tours.
- En cas de manque de pertinence, l'app affiche un message clair (au lieu d'halluciner).
- Les sources affichees restent coherentes avec la reponse.

## 8. Depannage

- Si le LLM ne se charge pas: verifier le chemin du GGUF dans `app_streamlit.py`.
- Si les resultats sont faibles: verifier la qualite des colonnes `Critere`, `Zone`, `Periode`.
- Si Excel est ouvert: fermer le fichier XLSB puis relancer.
