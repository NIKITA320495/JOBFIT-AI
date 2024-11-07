import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pdfplumber
from io import BytesIO
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
model = SentenceTransformer('all-MiniLM-L6-v2')

##### Extracting text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(BytesIO(pdf_file.read())) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

###### Matching keywords between resume and job description using Sentence-BERT (SBERT) embeddings
def keyword_match(job_description, user_profile):
    job_description_embedding = model.encode(job_description, convert_to_tensor=True)
    user_profile_embedding = model.encode(user_profile, convert_to_tensor=True)

    # Compute similarity score
    similarity_score = util.cos_sim(job_description_embedding, user_profile_embedding).item()

    # Display the similarity score as a percentage
    return round(similarity_score * 100, 2)
# Load environment variables from .env file
load_dotenv()

#LLM1
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = 'true'

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are expert career coach and consultant. You will generate  proper cover letter based on user profile and user input."),
    ("user", "  From the {user_profile}, please create a cover letter for the {job_role} position at {company_name} company. Ensure that it is well-crafted and engaging for recruiters and hiring managers. Also, verify that my recent work experience and academic background align with the role I am applying for.")
])

# Initialize OpenAI model and parser
llm = ChatOpenAI(api_key=openai_api_key)
output_parser = StrOutputParser()
chain=prompt|llm|output_parser

####################LLM2
prompt2 = ChatPromptTemplate.from_messages([
    ("system", "You are a skilled Application Tracking System (ATS) expert with a deep understanding of the tech field, software engineering, data science, data analysis and big data engineering. Your task is to evaluate the resume based on the given resume and job description."),
    ("user", "  From the {user_profile} and {job_desc}, please evaluate the resume and provide a descriptive feedback on how well the resume aligns with the job description. Also, suggest any improvements that can be made to increase the chances of getting shortlisted for the job.")
])

# Initialize OpenAI model and parser
llm2 = ChatOpenAI(api_key=openai_api_key)
output_parser2 = StrOutputParser()
chain2=prompt2|llm2|output_parser2
##################Linkedin Jobs####################

def job_finder(user_input_job_title, job_location):
    job_title_query = ', '.join(user_input_job_title).replace(' ', '%20')

    link = f"https://in.linkedin.com/jobs/search?keywords={job_title_query}&location={job_location}&geoId=102713980&f_TPR=r604800&position=1&pageNum=0"
    
    driver = webdriver.Chrome()
    driver.maximize_window()

    try:
        driver.get(link)

        # Explicit wait for job listings to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "h3.base-search-card__title"))
        )

        # Scroll to load more jobs
        for _ in range(2):  # Adjust this range as needed
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)  # Adjust sleep time as necessary

        # Click 'See more jobs' if available
        try:
            more_jobs_button = driver.find_element(by=By.CSS_SELECTOR, value="button[aria-label='See more jobs']")
            more_jobs_button.click()
            time.sleep(3)
        except Exception as e:
            print("No 'See more jobs' button found:", e)

        # Initialize lists to store job data
        company_name = []
        job_title = []
        company_location = []
        job_url = []

        # Extract job data
        companies = driver.find_elements(by=By.CSS_SELECTOR, value='h4.base-search-card__subtitle')
        titles = driver.find_elements(by=By.CSS_SELECTOR, value='h3.base-search-card__title')
        locations = driver.find_elements(by=By.CSS_SELECTOR, value='span.job-search-card__location')
        urls = driver.find_elements(by=By.XPATH, value='//a[contains(@href, "/jobs/")]')

        # Limit to top 5 jobs
        for i in range(min(5, len(companies))):
            company_name.append(companies[i].text)
            job_title.append(titles[i].text)
            company_location.append(locations[i].text)
            job_url.append(urls[i].get_attribute('href'))

        # Create DataFrame to store job listings
        df = pd.DataFrame({
            'Company Name': company_name,
            'Job Title': job_title,
            'Location': company_location,
            'Website URL': job_url
        })
        return df

    except Exception as e:
        print("An error occurred:", e)
        return None

    finally:
        # Close the driver
        driver.quit()
##################INTERVIEW PREPARATION####################
prompt3 = ChatPromptTemplate.from_messages([
    ("system", "You are a skilled interview coach with expertise in the tech fields. Your task is to provide relevant questions and answers for the user to prepare for the interview according to difficulty level."),
    ("user", "  From the {job_role}, {description} and {difficulty_level}, please provide relevant questions and answers for the user to prepare for the interview.")
])
# Initialize OpenAI model and parser
llm3 = ChatOpenAI(api_key=openai_api_key)
output_parser3 = StrOutputParser()
chain3=prompt3|llm3|output_parser3

######Streamlit interface####################
st.sidebar.title("JOBFit AI")
option = st.sidebar.selectbox("Select an option", ["Resume Optimizer", "Personalized CV Creator","Linkedin Jobs","Interview Preparation"])
# Displaying each option as a list item using Markdown

if option =='Resume Optimizer':
    st.title("Resume Optimizer")
    profile = st.file_uploader("Upload a PDF file of your resume", type=["pdf"])
    job_desc = st.text_area("Enter the job description you are applying for:")

    # Process only if profile is uploaded
    if profile is not None:
        # Extract text from uploaded PDF
        user_profile = extract_text_from_pdf(profile)

        # Check if both user_profile and job description are provided
        if user_profile and job_desc:
            # Calculate keyword match score (use your actual logic here)
            match = keyword_match(user_profile, job_desc)
            st.write("Keyword match between resume and job description:", match, '%')

            # Example placeholder for LLM chain invocation
            # Replace chain2 with your actual chain object
            # st.write(chain2.invoke({'user_profile': user_profile, 'job_desc': job_desc}))
        else:
            st.warning("Please enter the job description to evaluate the resume.")
    else:
        st.info("Please upload your Resume to get started.")

elif option =='Personalized CV Creator':
    st.title("Personalized Cover Letter Creator")
    job_role = st.text_input("Enter the job role you are applying for:", placeholder="e.g., Data Scientist")
    company_name = st.text_input("Enter the company name you are applying to:", placeholder="e.g., LangSmith")
    profile = st.file_uploader("Upload a PDF file of your profile", type=["pdf"])
    user_profile=""
    if profile is not None:
        user_profile = extract_text_from_pdf(profile)
    if job_role and user_profile:
        st.write(chain.invoke({'job_role': job_role, 'user_profile': user_profile}))
    elif profile is not None:
        st.warning("Please enter the job role to generate the cover letter.")
    else:
        st.info("Please upload your profile document to get started.")

elif option =='Linkedin Jobs':
    st.title("Linkedin Jobs")
    job_input=st.text_input("Enter the job role you are looking for:", placeholder="e.g., Data Scientist")
    location_input=st.text_input("Enter the location you are looking for:", placeholder="e.g., New Delhi")
    if job_input and location_input:
        st.write("Here are some job listings for", job_input, "in", location_input)
        jobs_df = job_finder(job_input,location_input)
        print(jobs_df)
    else:
        st.info("Please enter the job role and location to get job listings.")

elif option =='Interview Preparation':
    st.title("Interview Preparation")
    job_input=st.text_input("Enter the job role", placeholder="e.g., Data Scientist")
    description=st.text_area("Enter the job description")
    difficulty_level=st.selectbox("Select the difficulty level", ["Easy", "Medium", "Hard"])
    if job_input and description and difficulty_level:
        st.write(chain3.invoke({'job_role': job_input, 'description': description, 'difficulty_level': difficulty_level}))
    else:
        st.info("Please enter the job role, job description and difficulty level to get interview questions.")