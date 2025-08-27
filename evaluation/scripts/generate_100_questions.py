#!/usr/bin/env python3
"""
Generate 100 High-Quality Questions for RAG Evaluation
Selects diverse missions and generates targeted questions
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from tqdm import tqdm
import asyncio
import hashlib

from llama_index.llms.openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Focused100QuestionGenerator:
    """Generates 100 high-quality questions from diverse missions"""
    
    def __init__(
        self,
        documents_dir: str = None,
        output_dir: str = None,
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.1
    ):
        """Initialize the generator"""
        # Find the project root directory
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent  # Go up to SpaceMissionDesignAssistant
        
        # Set default paths relative to project root
        if documents_dir is None:
            documents_dir = project_root / "rag_ready_data" / "combined_documents"
        if output_dir is None:
            output_dir = script_dir.parent / "data"
            
        self.documents_dir = Path(documents_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Documents directory: {self.documents_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        
        self.llm = OpenAI(model=llm_model, temperature=temperature)
        
        # Question distribution for 100 questions
        self.question_distribution = {
            "factual": 20,      # Basic facts
            "technical": 20,    # Technical specifications
            "comparison": 10,   # Comparing missions/components
            "temporal": 10,     # Timeline/dates
            "analytical": 15,   # Analysis and reasoning
            "multi_hop": 10,    # Combining information
            "numerical": 10,    # Numbers and calculations
            "causal": 5         # Cause-effect relationships
        }
        
    def select_diverse_missions(self, num_missions: int = 20) -> List[Dict[str, Any]]:
        """Select diverse missions for question generation"""
        all_files = list(self.documents_dir.glob("*_combined.json"))
        
        if not all_files:
            logger.error(f"No mission files found in {self.documents_dir}")
            return []
            
        # Randomly select diverse missions
        selected_files = random.sample(all_files, min(num_missions, len(all_files)))
        
        missions = []
        for file_path in selected_files:
            try:
                with open(file_path, 'r') as f:
                    mission_data = json.load(f)
                    missions.append(mission_data)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                
        logger.info(f"Selected {len(missions)} diverse missions")
        return missions
    
    async def generate_questions_for_mission(
        self,
        mission: Dict[str, Any],
        question_type: str,
        num_questions: int
    ) -> List[Dict[str, Any]]:
        """Generate specific questions for a mission"""
        
        questions = []
        
        # Extract key information
        mission_id = mission.get('mission_id', 'unknown')
        mission_text = mission.get('text', '')[:3000]  # First 3000 chars
        
        # Create focused prompt
        prompt = f"""Generate {num_questions} {question_type} question(s) about the {mission_id} space mission.

Mission Information:
{mission_text}

Question Type: {question_type}
- factual: Ask about specific facts, components, or specifications
- technical: Ask about technical details, parameters, or engineering aspects  
- comparison: Compare aspects within this mission or with typical standards
- temporal: Ask about launch dates, mission phases, or timelines
- analytical: Ask about mission objectives, challenges, or impacts
- multi_hop: Questions requiring combining multiple pieces of information
- numerical: Questions about numbers, measurements, or calculations
- causal: Questions about cause-effect relationships or reasons

For each question, provide:
1. A clear, answerable question
2. The correct answer based on the document
3. Relevant text excerpts that support the answer

Return as JSON array:
[
  {{
    "question": "What is...",
    "answer": "The answer is...",
    "context": "From the document: ...",
    "keywords": ["keyword1", "keyword2"],
    "difficulty": "easy|medium|hard"
  }}
]"""

        try:
            response = await self.llm.acomplete(prompt)
            
            # Parse response
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
                
            generated = json.loads(response_text)
            
            # Add metadata
            for q in generated:
                q['mission_id'] = mission_id
                q['question_type'] = question_type
                q['question_id'] = hashlib.md5(
                    f"{mission_id}_{q['question']}".encode()
                ).hexdigest()[:8]
                q['relevant_docs'] = [mission_id]
                questions.append(q)
                
        except Exception as e:
            logger.error(f"Error generating {question_type} questions for {mission_id}: {e}")
            
        return questions
    
    async def generate_100_questions(self):
        """Generate exactly 100 high-quality questions"""
        logger.info("Starting generation of 100 questions...")
        
        # Select diverse missions
        missions = self.select_diverse_missions(20)
        
        if not missions:
            logger.error("No missions found!")
            return []
        
        all_questions = []
        
        # Generate questions by type
        for question_type, count in self.question_distribution.items():
            logger.info(f"Generating {count} {question_type} questions...")
            
            questions_generated = 0
            mission_idx = 0
            
            while questions_generated < count and mission_idx < len(missions):
                # Questions per mission for this type
                questions_needed = min(3, count - questions_generated)
                
                mission = missions[mission_idx % len(missions)]
                
                # Generate questions
                questions = await self.generate_questions_for_mission(
                    mission,
                    question_type,
                    questions_needed
                )
                
                all_questions.extend(questions)
                questions_generated += len(questions)
                mission_idx += 1
                
                # Small delay to respect rate limits
                await asyncio.sleep(0.5)
        
        logger.info(f"Generated {len(all_questions)} total questions")
        
        # Save dataset
        self._save_dataset(all_questions)
        
        return all_questions
    
    def _save_dataset(self, questions: List[Dict[str, Any]]):
        """Save the 100-question dataset"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main dataset
        output_path = self.output_dir / f"eval_100_questions_{timestamp}.json"
        
        dataset = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_questions': len(questions),
                'question_distribution': self.question_distribution,
                'description': 'Focused evaluation dataset with 100 high-quality questions'
            },
            'questions': questions
        }
        
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
            
        logger.info(f"Saved 100-question dataset to {output_path}")
        
        # Also save by type for analysis
        for q_type in self.question_distribution.keys():
            typed_questions = [q for q in questions if q['question_type'] == q_type]
            if typed_questions:
                type_path = self.output_dir / f"eval_100_{q_type}_{timestamp}.json"
                with open(type_path, 'w') as f:
                    json.dump(typed_questions, f, indent=2)


async def main():
    """Generate 100 evaluation questions"""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    # Create generator (paths will be auto-detected)
    generator = Focused100QuestionGenerator(
        llm_model="gpt-4o-mini",
        temperature=0.1
    )
    
    # Generate 100 questions
    questions = await generator.generate_100_questions()
    
    if questions:
        print("\n✓ Successfully generated 100 evaluation questions!")
        print(f"✓ Questions cover {len(generator.question_distribution)} different types")
        print("✓ Dataset saved to: evaluation/data/eval_100_questions_*.json")
    else:
        print("✗ Failed to generate questions")


if __name__ == "__main__":
    asyncio.run(main())