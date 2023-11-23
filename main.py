import numpy as np
import pickle
import streamlit as st

st.write(f"scikit-learn version: {scikit-learn.__version__}")
# Charger le modèle
model = pickle.load(open('./modele.pkl', 'rb'))

# Page d'accueil
def home():
    st.title("Votre Application Streamlit")
    # Utilisez st.write pour afficher des textes, des instructions, etc.
    st.write("Remplissez le formulaire ci-dessous pour obtenir une prédiction.")

    # Interface utilisateur pour la saisie des valeurs
    checking_status = st.text_input("Statut du compte", "")
    duration = st.text_input("Durée du crédit", "")
    credit_history = st.text_input("Antécédents de crédit", "")
    purpose = st.text_input("Objet du crédit", "")
    credit_amount = st.text_input("Montant du crédit", "")
    savings_status = st.text_input("Situation du compte d'épargne/des obligations", "")
    employment = st.text_input("Emploi actuel, en nombre d'années.", "")
    installment_commitment = st.text_input("Taux de remboursement en pourcentage du revenu disponible", "")
    other_parties = st.text_input("Autres débiteurs / garants", "")
    residence_since = st.text_input("Résidence actuelle depuis X années", "")
    property_magnitude = st.text_input("Biens (par exemple, biens immobiliers)", "")
    age = st.text_input("Age", "")
    other_payment_plans = st.text_input("Autres plans de paiement", "")
    housing = st.text_input("Logement (location, propriété,...)", "")
    num_dependents = st.text_input("Nombre de personnes à charge", "")
    own_telephone = st.text_input("Téléphone", "")
    foreign_worker = st.text_input("Travailleur étranger", "")
    sex = st.text_input("Genre", "")
    marriage = st.text_input("Statut matrimonial", "")

    # Utilisez st.button pour le bouton de prédiction
    if st.button("Prédire"):
        # Faire la prédiction
        input_features = np.array([float(checking_status), float(duration), float(credit_history), float(purpose),
                                   float(credit_amount), float(savings_status), float(employment),
                                   float(installment_commitment), float(other_parties), float(residence_since),
                                   float(property_magnitude), float(age), float(other_payment_plans), float(housing),
                                   float(num_dependents), float(own_telephone), float(foreign_worker), float(sex),
                                   float(marriage)]).reshape(1, -1)
        prediction = model.predict(input_features)

        # Afficher le résultat
        st.success(f"Ce client est-il à risque ? {round(prediction[0], 2)}")

if __name__ == "__main__":
    home()
