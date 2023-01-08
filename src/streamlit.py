import streamlit as st
import requests
from PIL import Image

# Load and set images in the first place
header_images = Image.open('assets/header_images.jpg')
st.image(header_images)

# Add some information about the service
st.title("Auto Insurance Fraud Detection")
st.subheader("Just enter variabel below then click Predict button :sunglasses:")

# Create form of input
with st.form(key = "auto_insurance_data_form"):
    # Create box for number input
    months_as_customer = st.number_input(
        label = "2.\tEnter months_as_customer Value:",
        min_value = -1,
        max_value = 500,
        help = "Value range from 0 to 500"
    )

    age = st.number_input(
        label = "2.\tEnter age Value:",
        min_value = 19,
        max_value = 65,
        help = "Value range from 19 to 65"
    )

    policy_number = st.number_input(
        label = "2.\tEnter policy_number Value:",
        min_value = 100000,
        max_value = 999999,
        help = "Value range from 100000 to 999999"
    )

    policy_annual_premium = st.number_input(
        label = "2.\tEnter policy_annual_premium Value:",
        min_value = 400,
        max_value = 2500,
        help = "Value range from 400 to 2500"
    )

    insured_zip = st.number_input(
        label = "2.\tEnter insured_zip Value:",
        min_value = 430000,
        max_value = 630000,
        help = "Value range from 430000 to 630000"
    )

    capital_gains = st.number_input(
        label = "2.\tEnter capital-gains Value:",
        min_value = 0,
        max_value = 150000,
        help = "Value range from 0 to 150000"
    )

    capital_loss = st.number_input(
        label = "2.\tEnter capital-loss Value:",
        min_value = -120000,
        max_value = 0,
        help = "Value range from -120000 to 0"
    )

    incident_hour_of_the_day = st.number_input(
        label = "2.\tEnter incident_hour_of_the_day Value:",
        min_value = -1,
        max_value = 23,
        help = "Value range from 0 to 23"
    )

    total_claim_amount = st.number_input(
        label = "2.\tEnter total_claim_amount Value:",
        min_value = 99,
        max_value = 120000,
        help = "Value range from 99 to 120000"
    )

    injury_claim = st.number_input(
        label = "2.\tEnter injury_claim Value:",
        min_value = -1,
        max_value = 25000,
        help = "Value range from 0 to 120000"
    )

    property_claim = st.number_input(
        label = "2.\tEnter property_claim Value:",
        min_value = -1,
        max_value = 25000,
        help = "Value range from 0 to 120000"
    )

    vehicle_claim = st.number_input(
        label = "2.\tEnter vehicle_claim Value:",
        min_value = 0,
        max_value = 80000,
        help = "Value range from 0 to 80000"
    )

    # Create select box input
    policy_deductable = st.selectbox(
        label = "1.\tFrom which policy_deductable is this data collected?",
        options = (
            "500",
            "1000",
            "2000"
        )
    )

    umbrella_limit = st.selectbox(
        label = "1.\tFrom which umbrella_limit is this data collected?",
        options = (
            "0",
            "1000000",
            "2000000",
            "3000000",
            "4000000",
            "5000000",
            "6000000",
            "7000000",
            "8000000",
            "9000000",
            "10000000"
        )
    )

    number_of_vehicles_involved = st.selectbox(
        label = "1.\tFrom which number_of_vehicles_involved is this data collected?",
        options = (
            "1",
            "2",
            "3",
            "4"
        )
    )

    bodily_injuries = st.selectbox(
        label = "1.\tFrom which bodily_injuries is this data collected?",
        options = (
            "0",
            "1",
            "2"
        )
    )

    witnesses = st.selectbox(
        label = "1.\tFrom which witnesses is this data collected?",
        options = (
            "0",
            "1",
            "2",
            "3"
        )
    )

    auto_year = st.selectbox(
        label = "1.\tFrom which auto_year is this data collected?",
        options = (
            "1995",
            "1999",
            "2005",
            "2006",
            "2011",
            "2007",
            "2003",
            "2009",
            "2010",
            "2013",
            "2002",
            "2015",
            "1997",
            "2012",
            "2008",
            "2014",
            "2001",
            "2000",
            "1998",
            "2004",
            "1996"
        )
    )

    policy_state = st.selectbox(
        label = "1.\tFrom which policy_state is this data collected?",
        options = (
            "OH",
            "IL",
            "IN"
        )
    )

    policy_csl = st.selectbox(
        label = "1.\tFrom which policy_csl is this data collected?",
        options = (
            "250/500",
            "100/300",
            "500/1000"
        )
    )

    insured_sex = st.selectbox(
        label = "1.\tFrom which insured_sex is this data collected?",
        options = (
            "FEMALE",
            "MALE"
        )
    )

    insured_hobbies = st.selectbox(
        label = "1.\tFrom which insured_hobbies is this data collected?",
        options = (
            "reading",
            "exercise",
            "paintball",
            "bungie-jumping",
            "movies",
            "golf",
            "camping",
            "kayaking",
            "yachting",
            "hiking",
            "video-games",
            "skydiving",
            "base-jumping",
            "board-games",
            "polo",
            "chess",
            "dancing",
            "sleeping",
            "cross-fit",
            "basketball"
        )
    )

    incident_type = st.selectbox(
        label = "1.\tFrom which incident_type is this data collected?",
        options = (
            "Multi-vehicle Collision",
            "Single Vehicle Collision",
            "Vehicle Theft",
            "Parked Car"
        )
    )

    collision_type = st.selectbox(
        label = "1.\tFrom which collision_type is this data collected?",
        options = (
            "Rear Collision",
            "Side Collision",
            "Front Collision",
            "UNKNOWN"
        )
    )

    incident_severity = st.selectbox(
        label = "1.\tFrom which incident_severity is this data collected?",
        options = (
            "Minor Damage",
            "Total Loss",
            "Major Damage",
            "Trivial Damage"
        )
    )

    authorities_contacted = st.selectbox(
        label = "1.\tFrom which authorities_contacted is this data collected?",
        options = (
            "Police",
            "Fire",
            "Other",
            "Ambulance",
            "None "
        )
    )

    incident_state = st.selectbox(
        label = "1.\tFrom which incident_state is this data collected?",
        options = (
            "NY",
            "SC",
            "WV",
            "VA",
            "NC",
            "PA",
            "OH"
        )
    )

    incident_city = st.selectbox(
        label = "1.\tFrom which incident_city is this data collected?",
        options = (
            "Springfield",
            "Arlington",
            "Columbus",
            "Northbend",
            "Hillsdale",
            "Riverwood",
            "Northbrook"
        )
    )

    property_damage = st.selectbox(
        label = "1.\tFrom which property_damage is this data collected?",
        options = (
            "UNKNOWN",
            "NO",
            "YES"
        )
    )

    police_report_available = st.selectbox(
        label = "1.\tFrom which police_report_available is this data collected?",
        options = (
            "UNKNOWN",
            "NO",
            "YES"
        )
    )

    auto_make = st.selectbox(
        label = "1.\tFrom which auto_make is this data collected?",
        options = (
            "Saab",
            "Dodge",
            "Suburu",
            "Nissan",
            "Chevrolet",
            "Ford",
            "BMW",
            "Toyota",
            "Audi",
            "Accura",
            "Volkswagen",
            "Jeep",
            "Mercedes",
            "Honda"
        )
    )

    auto_model = st.selectbox(
        label = "1.\tFrom which auto_model is this data collected?",
        options = (
            "RAM",
            "Wrangler",
            "A3",
            "Neon",
            "MDX",
            "Jetta",
            "Passat",
            "A5",
            "Legacy",
            "Pathfinder",
            "Malibu",
            "92x",
            "Camry",
            "Forrestor",
            "F150",
            "95",
            "E400",
            "93",
            "Grand",
            "Escape",
            "Tahoe",
            "Maxima",
            "Ultima",
            "X5",
            "Highlander",
            "Civic",
            "Silverado",
            "Fusion",
            "ML350",
            "Impreza",
            "Corolla",
            "TL",
            "CRV",
            "C300",
            "3",
            "X6",
            "M5",
            "Accord",
            "RSX"
        )
    )

    # Create button to submit the form
    submitted = st.form_submit_button("Predict")

    # Condition when form submitted
    if submitted:
        # Create dict of all data in the form
        raw_data = {
            "months_as_customer": months_as_customer,
            "age": age,
            "policy_number": policy_number,
            "policy_deductable": policy_deductable,
            "policy_annual_premium": policy_annual_premium,
            "umbrella_limit": umbrella_limit,
            "insured_zip": insured_zip,
            "capital-gains": capital_gains,
            "capital-loss": capital_loss,
            "incident_hour_of_the_day": incident_hour_of_the_day,
            "number_of_vehicles_involved": number_of_vehicles_involved,
            "bodily_injuries": bodily_injuries,
            "witnesses": witnesses,
            "total_claim_amount": total_claim_amount,
            "injury_claim": injury_claim,
            "property_claim": property_claim,
            "vehicle_claim": vehicle_claim,
            "auto_year": auto_year,
            "policy_state": policy_state,
            "policy_csl": policy_csl,
            "insured_sex": insured_sex,
            "insured_hobbies": insured_hobbies,
            "incident_type": incident_type,
            "collision_type": collision_type,
            "incident_severity": incident_severity,
            "authorities_contacted": authorities_contacted,
            "incident_state": incident_state,
            "incident_city": incident_city,
            "property_damage": property_damage,
            "police_report_available": police_report_available,
            "auto_make": auto_make,
            "auto_model": auto_model        
        }

        # Create loading animation while predicting
        with st.spinner("Sending data to prediction server ..."):
        res = requests.post("http://api:8080/predict", json = raw_data).json()
        
        # Parse the prediction result
        if res["error_msg"] != "":
            st.error("Error Occurs While Predicting: {}".format(res["error_msg"]))
        else:
            if res["res"] != "FRAUD":
                st.warning("Insurance Claim Predicted: FRAUD.")
            else:
                st.success("Insurance Claim Predicted: TIDAK FRAUD.")