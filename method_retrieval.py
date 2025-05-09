import os
import json
import logging
from typing import Dict, List, Any, Optional
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MechanicalDesignRetrievalSystem:
    """Mechanical concept design retrieval system"""

    def __init__(self, config_path: str = "config.json"):
        """Initialize the retrieval system"""
        self.config = self._load_config(config_path)
        self.llm = OpenAI(
            temperature=self.config.get("temperature", 0.1),
            model_name=self.config.get("model_name", "gpt-4"),
            max_tokens=self.config.get("max_tokens", 4000)
        )
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self.qa_chain = None
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load the configuration file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Configuration file {config_path} not found, using default configuration")
            return {}

    def process_design_data(self, data_dir: str) -> None:
        """Process design data and create a vector store"""
        # Load different types of design data
        loaders = []

        # Load requirement analysis documents
        if os.path.exists(os.path.join(data_dir, "requirements")):
            req_loader = DirectoryLoader(
                os.path.join(data_dir, "requirements"),
                glob="**/*.txt",
                loader_cls=TextLoader
            )
            loaders.append(req_loader)

        # Load design text content
        if os.path.exists(os.path.join(data_dir, "design_texts")):
            text_loader = DirectoryLoader(
                os.path.join(data_dir, "design_texts"),
                glob="**/*.txt",
                loader_cls=TextLoader
            )
            loaders.append(text_loader)

        # Load design drawing metadata (assuming there are text description files for design drawings)
        if os.path.exists(os.path.join(data_dir, "design_drawings")):
            drawing_loader = DirectoryLoader(
                os.path.join(data_dir, "design_drawings"),
                glob="**/*.txt",  # Text descriptions of design drawings
                loader_cls=TextLoader
            )
            loaders.append(drawing_loader)

        # Load historical cases
        if os.path.exists(os.path.join(data_dir, "historical_cases")):
            case_loader = DirectoryLoader(
                os.path.join(data_dir, "historical_cases"),
                glob="**/*.pdf",
                loader_cls=PyPDFLoader
            )
            loaders.append(case_loader)

        # Read all documents
        documents = []
        for loader in loaders:
            documents.extend(loader.load())

        # Text splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.get("chunk_size", 1000),
            chunk_overlap=self.config.get("chunk_overlap", 200)
        )
        split_docs = text_splitter.split_documents(documents)

        # Create a vector store
        logger.info(f"Creating a vector store with {len(split_docs)} document chunks")
        self.vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            persist_directory=self.config.get("persist_directory", "db")
        )

        # Define the prompt template
        prompt_template = """
        You are a professional mechanical concept design assistant. Based on the user's design requirements, provide the most suitable design scheme suggestions from the historical case library and mechanical engineering knowledge base. The response should include the following parts:

        1. Requirement analysis: Analyze the key functions, performance indicators, and constraints of the user's requirements.
        2. Historical reusable cases: Recommend 3 - 5 most relevant historical design cases, including case names, main features, and applicable scenarios.
        3. Global reference cases: Provide 1 - 3 innovative international cases of similar designs.
        4. Design text content: Provide detailed design suggestions based on historical cases and the latest technologies.
        5. Reference schematic diagrams: Briefly describe the types of schematic diagrams that can support the design (such as assembly diagrams, schematic diagrams, etc.).

        Design requirements: {question}
        Historical conversation: {chat_history}
        """

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["question", "chat_history"]
        )

        # Create a retrieval chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            self.llm,
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": self.config.get("search_k", 10)}
            ),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )

    def analyze_user_requirements(self, requirements: str) -> Dict[str, Any]:
        """Analyze user requirements and extract key information"""
        analysis_prompt = f"""
        Please analyze the following mechanical design requirements and extract key functions, performance indicators, constraints, and design objectives:

        Requirement description: {requirements}

        Please return the analysis result in JSON format, including the following fields:
        - key_functions: List of key functions
        - performance_metrics: List of performance indicators
        - constraints: List of constraints
        - design_objectives: List of design objectives
        """

        try:
            response = self.llm(analysis_prompt)
            # Try to parse as JSON
            analysis_result = json.loads(response)
            return analysis_result
        except json.JSONDecodeError:
            logger.error(f"Failed to parse requirement analysis result: {response}")
            # Return a structured default result
            return {
                "key_functions": [],
                "performance_metrics": [],
                "constraints": [],
                "design_objectives": []
            }

    def generate_design_solutions(self, requirements: str) -> Dict[str, Any]:
        """Generate design solutions"""
        if not self.qa_chain:
            raise ValueError("Please process the design data and initialize the retrieval system first")

        logger.info(f"Generating design solutions: {requirements[:50]}...")
        result = self.qa_chain({"question": requirements})

        # Try to parse as a structured result
        try:
            structured_result = json.loads(result["answer"])
        except json.JSONDecodeError:
            # If it cannot be parsed as JSON, keep the original text format
            structured_result = {
                "raw_answer": result["answer"],
                "source_documents": [doc.metadata for doc in result["source_documents"]]
            }

        return structured_result

    def evaluate_design_solution(self, solution: Dict[str, Any], criteria: List[str]) -> Dict[str, float]:
        """Evaluate the design solution"""
        evaluation_prompt = f"""
        Please evaluate the mechanical design solution according to the following criteria:

        Evaluation criteria: {', '.join(criteria)}

        Design solution: {json.dumps(solution, indent=2)}

        Please return the score (0 - 10 points) for each criterion and an overall evaluation.
        """

        response = self.llm(evaluation_prompt)

        # Parse the evaluation result
        try:
            evaluation_result = json.loads(response)
            return evaluation_result
        except json.JSONDecodeError:
            logger.error(f"Failed to parse the evaluation result: {response}")
            return {"raw_evaluation": response}

    def refine_design_solution(self, solution: Dict[str, Any], feedback: str) -> Dict[str, Any]:
        """Refine the design solution based on feedback"""
        refinement_prompt = f"""
        Please refine the following mechanical design solution based on user feedback:

        Original solution: {json.dumps(solution, indent=2)}

        User feedback: {feedback}

        Please provide the refined design scheme, keeping the same JSON structure.
        """

        response = self.llm(refinement_prompt)

        try:
            refined_solution = json.loads(response)
            return refined_solution
        except json.JSONDecodeError:
            logger.error(f"Failed to parse the refined solution: {response}")
            return {"raw_refined_solution": response}


# API model definition
class RequirementAnalysisRequest(BaseModel):
    requirements: str


class DesignSolutionRequest(BaseModel):
    requirements: str


class SolutionEvaluationRequest(BaseModel):
    solution: Dict[str, Any]
    criteria: List[str]


class SolutionRefinementRequest(BaseModel):
    solution: Dict[str, Any]
    feedback: str


# Create a FastAPI application
app = FastAPI(title="Mechanical Concept Design Assistant")

# Mount the static file directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Template engine
templates = Jinja2Templates(directory="templates")

# Initialize the retrieval system
retrieval_system = MechanicalDesignRetrievalSystem()


# API endpoints
@app.on_event("startup")
async def startup_event():
    try:
        # Process design data
        retrieval_system.process_design_data("design_data")
        logger.info("System startup completed, design data loaded")
    except Exception as e:
        logger.error(f"System startup failed: {str(e)}")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/analyze_requirements")
async def analyze_requirements(request: RequirementAnalysisRequest):
    try:
        result = retrieval_system.analyze_user_requirements(request.requirements)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate_solution")
async def generate_solution(request: DesignSolutionRequest):
    try:
        result = retrieval_system.generate_design_solutions(request.requirements)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/evaluate_solution")
async def evaluate_solution(request: SolutionEvaluationRequest):
    try:
        result = retrieval_system.evaluate_design_solution(request.solution, request.criteria)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/refine_solution")
async def refine_solution(request: SolutionRefinementRequest):
    try:
        result = retrieval_system.refine_design_solution(request.solution, request.feedback)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    