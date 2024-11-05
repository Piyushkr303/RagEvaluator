import streamlit as st
import os
import PyPDF2
import logging
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from typing import Dict, List, Tuple
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEvaluator:
    def __init__(self):
        self.groq_api_key = None
        self.llm = None
        self.evaluation_prompt = None
        self.load_environment()
        self.setup_llm()
        self.setup_prompts()

    def load_environment(self):
        """Load environment variables and validate API key."""
        load_dotenv()
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("Groq API Key is missing. Please check your environment variables.")

    def setup_llm(self):
        """Initialize the LLM with error handling."""
        try:
            self.llm = ChatGroq(
                api_key=self.groq_api_key,
                model_name="mixtral-8x7b-32768",
                temperature=0.1,  # Reduced further for more consistent scoring
                max_tokens=4096
            )
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise

    def setup_prompts(self):

        self.evaluation_prompt = PromptTemplate(
        input_variables=["criteria", "text", "reference_solution"],
        template="""
        You are an extremely demanding evaluator who holds submissions to the highest possible standards. Your role is to meticulously analyze the submitted text against specific criteria, actively looking for flaws and areas for improvement. Full marks should be extremely rare and only awarded for truly exceptional implementations that go above and beyond requirements.
        Evaluation Instructions:
        1. Begin by looking for potential flaws and missing elements
        2. Scrutinize implementation quality, efficiency, and thoroughness
        3. Compare critically against the reference solution
        4. Always suggest improvements, even for good solutions
        5. Reserve perfect scores (10/10) only for solutions that are demonstrably superior to the reference
        Evaluation Criteria:
        {criteria}
        Reference Solution:
        {reference_solution}
        Text to evaluate:
        {text}
        For each criterion, follow this rigorous evaluation process:
        1. First, list all potential issues and missing elements
        2. Look for implementation flaws and inefficiencies
        3. Compare thoroughly against reference solution
        4. Identify opportunities for optimization
        5. Only after finding all possible issues, assign a score
        Return your evaluation in this exact JSON format:
        {{
            "evaluations": [
                {{
                    "criterion": "criterion_text",
                    "score": score_number,
                    "flaws_found": ["specific_flaw_1", "specific_flaw_2"],
                    "evidence_found": "Quote the specific evidence found in the text or state 'No evidence found'",
                    "justification": "Detailed explanation of why points were deducted, referencing specific issues",
                    "improvements": ["specific_improvement_1", "specific_improvement_2"],
                    "comparison": "critical_comparison_with_reference_highlighting_differences_and_shortcomings"
                }}
            ]
        }}
        Strict Scoring Guidelines:
        - Score 0-2: Minimal or flawed implementation
        - Score 3-4: Basic implementation with significant room for improvement
        - Score 5-6: Adequate implementation but with clear deficiencies
        - Score 7-8: Good implementation but still has minor issues or missing optimizations
        - Score 9: Excellent implementation matching reference solution completely
        - Score 10: Reserved ONLY for implementations that demonstrably improve upon the reference solution
        Deduction Requirements:
        - -1 point for each minor issue or inefficiency
        - -2 points for each significant flaw or missing feature
        - -3 points for major design issues or inefficient implementations
        - Additional deductions for any security concerns or potential edge cases not handled
        Before assigning a perfect score (10), verify that the solution:
        1. Implements everything in the reference solution
        2. Adds additional valuable features or optimizations
        3. Shows exceptional attention to detail
        4. Handles all edge cases comprehensively
        5. Demonstrates superior design choices
        Return only valid JSON without any additional text or explanations.
        """
    )

    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF with improved error handling."""
        if pdf_file is None:
            return ""
            
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = []
            for page in pdf_reader.pages:
                text.append(page.extract_text())
            return " ".join(text)
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            raise ValueError(f"Failed to process PDF file: {str(e)}")

    def evaluate_document(self, text: str, reference_text: str, criteria_list: List[str]) -> List[Dict]:
        """Evaluate document with strict scoring and validation."""
        try:
            if not text or not criteria_list:
                raise ValueError("Missing required input: text or criteria")

            criteria_text = "\n".join(f"{i+1}. {criterion}" for i, criterion in enumerate(criteria_list))
            
            formatted_prompt = self.evaluation_prompt.format(
                criteria=criteria_text,
                text=text[:10000],
                reference_solution=reference_text[:5000]
            )

            response = self.llm.predict(formatted_prompt)
            
            try:
                evaluation = json.loads(response)
                if "evaluations" not in evaluation:
                    raise ValueError("Invalid response format")
                return evaluation["evaluations"]
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON response: {response}")
                raise ValueError("Failed to parse evaluation results")
            
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            raise

def main():
    st.set_page_config(page_title="Document Review Tool", layout="wide")
    st.title("📝 Document Review and Grading Tool")
    st.write("Upload documents and enter evaluation criteria to get detailed feedback.")

    try:
        evaluator = RAGEvaluator()

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Student Submission")
            pdf_file = st.file_uploader("Upload PDF Document", type=['pdf'], key="submission")

        with col2:
            st.subheader("Reference Solution")
            reference_pdf = st.file_uploader("Upload Reference Solution", type=['pdf'], key="reference")
        
        criteria_text = st.text_area(
            "Enter Evaluation Criteria (one per line)", 
            height=200,
            value="Clear understanding and proper usage of data structures\n"
                  "Efficient implementation of algorithms\n"
                  "Code readability and proper documentation\n"
                  "Error handling and edge cases\n"
                  "Use of appropriate design patterns"
        )

        if st.button("Evaluate Document"):
            if not pdf_file:
                st.error("Please upload a document to evaluate.")
                return

            with st.spinner("Evaluating document..."):
                try:
                    document_text = evaluator.extract_text_from_pdf(pdf_file)
                    reference_text = evaluator.extract_text_from_pdf(reference_pdf) if reference_pdf else ""
                    
                    criteria_list = [line.strip() for line in criteria_text.split('\n') if line.strip()]
                    
                    if not criteria_list:
                        st.error("Please enter at least one evaluation criterion.")
                        return

                    evaluations = evaluator.evaluate_document(document_text, reference_text, criteria_list)
                    
                    st.subheader("Evaluation Results")
                    
                    total_score = sum(eval['score'] for eval in evaluations)
                    max_score = len(evaluations) * 10
                    
                    # Display overall score with clear explanation
                    st.metric(
                        "Overall Score", 
                        f"{total_score}/{max_score} ({(total_score/max_score)*100:.1f}%)",
                        help="Scores are only awarded when there is clear evidence of implementation"
                    )
                    
                    # Display individual evaluations with evidence and justification
                    for eval in evaluations:
                        with st.expander(f"{eval['criterion']} - Score: {eval['score']}/10"):
                            st.write("**Evidence Found:**", eval['evidence_found'])
                            st.write("**Score Justification:**", eval['justification'])
                            st.write("**Comparison to Reference:**", eval['comparison'])
                            st.write("**Required Improvements:**")
                            for imp in eval['improvements']:
                                st.write(f"- {imp}")

                except Exception as e:
                    st.error(f"An error occurred during evaluation: {str(e)}")
                    logger.exception("Evaluation failed")

    except Exception as e:
        st.error(f"Failed to initialize the application: {str(e)}")
        logger.exception("Application initialization failed")

if __name__ == "__main__":
    main()
