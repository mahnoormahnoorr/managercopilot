# pip install fastapi uvicorn langchain langchain-community langchain-openai chromadb tiktoken pydantic
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from operator import itemgetter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda

# --- load your existing Chroma collection ---
PERSIST_DIR = "docs/docs/chroma"
COLLECTION = "onboarding"
emb = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=emb, collection_name=COLLECTION)

def build_where_filter(role, level, hire_type):
    terms = []
    if role: terms.append({"dept": {"$in": [role, "all"]}})
    if level: terms.append({"level": {"$in": [level, "all"]}})
    if hire_type: terms.append({"hire": {"$in": [hire_type, "all"]}})
    if not terms: return {}
    return terms[0] if len(terms)==1 else {"$and": terms}

def get_retriever(role, level, hire_type):
    return vectordb.as_retriever(search_kwargs={"k": 6, "filter": build_where_filter(role, level, hire_type)})

def join(docs):
    return "\n\n---\n\n".join(d.page_content for d in docs)

# --- schema for output ---
from typing import List, Literal
from pydantic import BaseModel as PydModel

class ChecklistItem(PydModel):
    id: int; text: str; owner: str; due: str; status: Literal["open","blocked","done"]="open"
class ManagerChecklist(PydModel):
    role: str; level: str; hire_type: str
    title: str; overview: str
    assumptions: List[str]=[]; open_questions: List[str]=[]
    checklist: List[ChecklistItem]
    risks: List[str]=[]; dependencies: List[str]=[]; metrics: List[str]=[]
    references: List[str]=[]

parser = JsonOutputParser(pydantic_object=ManagerChecklist)
prompt = ChatPromptTemplate.from_messages([
    ("system","Using ONLY the provided context, create a practical onboarding checklist for role '{role}' and seniority '{level}'. Output valid JSON per schema."),
    ("human","Role: {role}\nSeniority: {level}\nHire type: {hire_type}\nTask: {task}\n\nContext:\n{context}\n\n{format_instructions}")
]).partial(format_instructions=parser.get_format_instructions())
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def build_chain(role, level, hire_type):
    retriever = get_retriever(role, level, hire_type)
    return ({
        "context": itemgetter("task") | retriever | RunnableLambda(join),
        "task": itemgetter("task"),
        "role": itemgetter("role"),
        "level": itemgetter("level"),
        "hire_type": itemgetter("hire_type"),
    } | prompt | llm | parser)

# --- FastAPI ---
app = FastAPI()
# allow your GitHub Pages origin here instead of "*" for production
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class GenerateReq(BaseModel):
    role: str; level: str; hire_type: str; task: str

@app.post("/api/generate")
def generate(req: GenerateReq):
    chain = build_chain(req.role, req.level, req.hire_type)
    result = chain.invoke({"role": req.role, "level": req.level, "hire_type": req.hire_type, "task": req.task})
    # attach references
    docs = get_retriever(req.role, req.level, req.hire_type).get_relevant_documents(req.task)
    refs = []
    for d in docs:
        s = d.metadata.get("source") or d.metadata.get("url") or d.metadata.get("file_path")
        if s and s not in refs: refs.append(s)
    result["references"] = refs
    return result
