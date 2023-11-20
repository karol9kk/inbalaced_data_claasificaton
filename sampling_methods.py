from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTENC
from imblearn.combine import SMOTETomek

def data_oversampling() -> tuple:
    name = "Oversampling"
    oversample = RandomOverSampler(sampling_strategy=0.7, random_state=42)
    return oversample, name

def data_undersampling() -> tuple:
    name = "Undersampling"
    undersample = RandomUnderSampler(sampling_strategy=0.47, random_state=42)
    return undersample, name

def data_smote() -> tuple:
    name = "SMOTE"
    smote = SMOTE(random_state=42)
    return smote, name

def data_adasyn() -> tuple:
    name = "ADASYN"
    adasyn = ADASYN(random_state=42)
    return adasyn, name

def data_smotenc() -> tuple:
    name = "SMOTENC"
    smotenc = SMOTENC(random_state=42)
    return smotenc, name

def data_smote_tomek() -> tuple:
    name = "SMOTEtomek"
    smote_tomek = SMOTETomek(random_state=42)
    return smote_tomek, name

