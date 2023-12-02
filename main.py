import numpy as np
import pickle
import streamlit as st

model = pickle.load(open('./modele.pkl', 'rb'))


def home():
    st.title("Analyse du risque de crédit à partir d’un dossier de prêt")
    st.write("Odilia T.N. SOME 204672")
    image_url = "./pret.png"
    st.image(image_url, use_column_width=True)
    st.write("Remplissez le formulaire ci-dessous pour obtenir une prédiction.")
    checking_status = st.selectbox("Statut du compte (['Pas de contrôle', '<0', '0<=X<200', '>=200'] ,  [0,1,2,3])", [0,1,2,3])
    duration = st.text_input("Durée du crédit(la durée du credit en mois)", "")
    credit_history = st.selectbox("Antécédents de crédit.(['crédit critique/autre crédit existant', 'retardé précédemment', 'payé existant', 'pas de crédit/tout payé', 'tout payé'], [0,1,2,2,2])", [0,1,2])
    purpose = st.selectbox("Objet du crédit(['entreprise', 'voiture neuve', 'voiture d'occasion', 'éducation', 'recyclage', 'autre', 'appareil ménager', 'radio/tv', 'mobilier/équipement', 'réparation'], [5,5,4,4,3,3,3,2,2,1])", [1, 2, 3, 4, 5])
    credit_amount = st.text_input("Montant du crédit", "")
    savings_status = st.selectbox("Situation du compte d'épargne/des obligations(['pas d'économies connues', '<100','100<=X<500','500<=X<1000','>=1000'], [0,1,2,3,4])", [0,1,2,3,4])
    employment = st.selectbox("Emploi actuel, en nombre d'années.(['chômeurs', '<1','1<=X<4','4<=X<7','>=7'], [0,1,2,3,4])", [0,1,2,3,4])
    installment_commitment = st.text_input("Taux de remboursement en pourcentage du revenu disponible", "")
    other_parties = st.selectbox("Autres débiteurs / garants", [0,1,2])
    residence_since = st.text_input("Résidence actuelle depuis X années", "")
    property_magnitude = st.selectbox("Biens (par exemple, biens immobiliers)", [0,1,2,3])
    age = st.text_input("Age", "")
    other_payment_plans = st.slider("Autres plans de paiement",0,1)
    housing = st.selectbox("Logement (location, propriété,...)", [0,1,2])
    credit_history_sec = st.text_input("Nombre de crédits existants dans cette banque", "")
    job = st.selectbox("Emploi", [0,1,2,3])
    num_dependents = st.text_input("Nombre de personnes à charge", "")
    own_telephone = st.slider("Téléphone", min_value=0, max_value=1, value=0, step=1)
    foreign_worker = st.slider("Travailleur étranger", min_value=0, max_value=1, value=0, step=1)
    sex = st.selectbox("Genre", [0,1])
    marriage = st.slider("Statut matrimonial", min_value=0, max_value=1, value=0, step=1)

    if st.button("Prédire"):
        try:
            input_features = np.array([
            float(checking_status), float(duration), float(credit_history), float(purpose),
            float(credit_amount), float(savings_status), float(employment),
            float(installment_commitment), float(other_parties), float(residence_since),
            float(property_magnitude), float(age), float(other_payment_plans), float(housing),
            float(credit_history_sec),  # Nouveau champ
            float(job),  # Nouveau champ
            float(num_dependents), float(own_telephone), float(foreign_worker), float(sex),
            float(marriage)
            ]).reshape(1, -1)
        
            prediction = model.predict(input_features)
            if round(prediction[0], 2) == 0:
                st.success(f"Ce client sera t'il solvable ? Non")
            else:
                st.success(f"Ce client sera t'il solvable ? Oui")
        except:
            st.write("Une erreur s'est produite, merci de verifier les données saisies")

if __name__ == "__main__":
    home()
