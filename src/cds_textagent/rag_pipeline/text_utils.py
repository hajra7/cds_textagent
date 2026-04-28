"""
RAG Pipeline - Text Utilities
Entity extraction and text processing for the RAG pipeline
"""
import json
import re
from typing import Dict, List, Optional
from openai import OpenAI
import os


def extract_entities_with_llm(case_text: str, model: str = "gpt-4o-mini") -> Dict[str, Optional[str]]:
    """
    Extract entities (age, gender, radiology_region) from case_text using LLM
    
    Args:
        case_text: The clinical case text
        model: OpenAI model to use
    
    Returns:
        Dict with extracted entities: {age, gender, radiology_region}
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    prompt = f"""Extract the following information from the clinical case text below:
1. Patient age (as a number, e.g., "45" or age range like "40-50")
2. Patient gender (male/female/unknown)
3. Radiology region (e.g., chest, abdomen, head, spine, pelvis, extremity)

Clinical case text:
{case_text}

Return ONLY a JSON object with keys: age, gender, radiology_region
If information is not found, use null.
Example: {{"age": "65", "gender": "male", "radiology_region": "chest"}}
"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=150
        )
        
        result_text = response.choices[0].message.content.strip()

        # Parse JSON response — remove markdown code blocks if present
        result_text = re.sub(r'```json\s*', '', result_text)
        result_text = re.sub(r'```\s*', '', result_text)
        
        entities = json.loads(result_text)
        
        return {
            'age': entities.get('age'),
            'gender': entities.get('gender'),
            'radiology_region': entities.get('radiology_region')
        }
        
    except Exception as e:
        print(f"Error extracting entities: {e}")
        return {'age': None, 'gender': None, 'radiology_region': None}


def extract_entities_with_regex(case_text: str) -> Dict[str, Optional[str]]:
    """
    Extract entities using regex patterns (faster but less accurate)
    
    Args:
        case_text: The clinical case text
    
    Returns:
        Dict with extracted entities
    """
    entities = {
        'age': None,
        'gender': None,
        'radiology_region': None
    }
    
    text_lower = case_text.lower()
    
    # Extract age
    age_patterns = [
        r'(\d+)\s*(?:year|yr|y\.?o\.?)',  # 65 year old, 45yo, etc.
        r'age[:\s]+(\d+)',
        r'(\d+)-year-old',
        r'(\d+)\s*y/?o',
    ]
    for pattern in age_patterns:
        match = re.search(pattern, text_lower)
        if match:
            entities['age'] = match.group(1)
            break
    
    # Extract gender
    if re.search(r'\b(male|man|boy)\b', text_lower) and not re.search(r'\bfemale\b', text_lower):
        entities['gender'] = 'male'
    elif re.search(r'\b(female|woman|girl)\b', text_lower):
        entities['gender'] = 'female'
    
    # Extract radiology region
    regions = [
        'chest', 'lung', 'thorax', 'thoracic',
        'abdomen', 'abdominal', 'pelvis', 'pelvic',
        'head', 'brain', 'skull', 'cranial',
        'spine', 'spinal', 'cervical', 'lumbar', 'thoracic spine',
        'extremity', 'arm', 'leg', 'hand', 'foot', 'knee', 'shoulder',
        'neck', 'cardiac', 'heart'
    ]
    
    for region in regions:
        if re.search(r'\b' + region + r'\b', text_lower):
            entities['radiology_region'] = region
            break
    
    return entities


def clean_text(text: str) -> str:
    """
    Clean and normalize text
    
    Args:
        text: Raw text
    
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract important keywords from text
    
    Args:
        text: Input text
        max_keywords: Maximum number of keywords
    
    Returns:
        List of keywords
    """
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were', 'been', 'be',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
        'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
    }
    
    # Tokenize and filter
    words = re.findall(r'\b\w+\b', text.lower())
    keywords = [w for w in words if w not in stop_words and len(w) > 3]
    
    # Get most frequent
    from collections import Counter
    word_counts = Counter(keywords)
    
    return [word for word, _ in word_counts.most_common(max_keywords)]
