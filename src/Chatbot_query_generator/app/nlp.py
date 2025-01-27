import logging
import re
from typing import Dict, List, Optional, Union
from pathlib import Path

import nltk
import numpy as np
import spacy
from langdetect import detect
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.language import Language
from spacy.tokens import Doc, Token
from textblob import TextBlob
from transformers import pipeline


class CustomSentencizer:
    """Custom sentence segmentation component for spaCy."""

    def __init__(self):
        pass

    def __call__(self, doc: Doc) -> Doc:
        for sent in doc.sents:
            for token in sent:
                if token.text in ['.', '!', '?']:
                    doc[token.i].is_sent_start = True
        return doc


class NLPProcessor:
    """Advanced NLP processor with multiple language support and error handling."""

    SUPPORTED_MODELS = {
        'it': 'it_core_news_sm',
        'en': 'en_core_web_sm',
        'fr': 'fr_core_news_sm',
        'es': 'es_core_news_sm'
    }

    def __init__(self, model_name: str = "it_core_news_sm", fallback_lang: str = "en"):
        """
        Initialize NLP processor with specified model and fallback options.

        Args:
            model_name (str): Primary spaCy model name to load
            fallback_lang (str): Fallback language if primary model fails
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.fallback_lang = fallback_lang
        self.is_fallback = False

        try:
            # Try loading the primary model
            self._load_primary_model()
        except OSError:
            # If primary model fails, try fallback
            self.logger.warning(f"Modello {model_name} non trovato, tentativo con modello {fallback_lang}")
            try:
                self._load_fallback_model()
            except Exception as e:
                self.logger.error(f"Errore nel caricamento del modello fallback: {str(e)}")
                raise

        try:
            self._initialize_components()
        except Exception as e:
            self.logger.error(f"Errore nell'inizializzazione dei componenti: {str(e)}")
            raise

    def _load_primary_model(self):
        """Load the primary spaCy model."""
        try:
            self.nlp = spacy.load(self.model_name)
            self.is_fallback = False
        except OSError as e:
            self.logger.error(f"Impossibile caricare il modello primario: {str(e)}")
            raise

    def _load_fallback_model(self):
        """Load the fallback spaCy model."""
        fallback_model = self.SUPPORTED_MODELS.get(self.fallback_lang)
        if not fallback_model:
            raise ValueError(f"Lingua fallback non supportata: {self.fallback_lang}")

        try:
            self.nlp = spacy.load(fallback_model)
            self.is_fallback = True
        except OSError as e:
            self.logger.error(f"Impossibile caricare il modello fallback: {str(e)}")
            raise

    def _initialize_components(self):
        """Initialize NLP components and models."""
        # Initialize transformers pipelines with error handling
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="dbmdz/bert-base-italian-uncased-sentiment"
            )
        except Exception as e:
            self.logger.warning(f"Errore nel caricamento dell'analizzatore sentimenti: {str(e)}")
            self.sentiment_analyzer = None

        try:
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn"
            )
        except Exception as e:
            self.logger.warning(f"Errore nel caricamento del summarizer: {str(e)}")
            self.summarizer = None

        try:
            self.translator = pipeline(
                "translation",
                model="Helsinki-NLP/opus-mt-it-en"
            )
        except Exception as e:
            self.logger.warning(f"Errore nel caricamento del traduttore: {str(e)}")
            self.translator = None

        try:
            self.question_answerer = pipeline(
                "question-answering",
                model="deepset/roberta-base-squad2"
            )
        except Exception as e:
            self.logger.warning(f"Errore nel caricamento del question answerer: {str(e)}")
            self.question_answerer = None

        # Download NLTK data with error handling
        self._download_nltk_data()

        # Setup custom components
        self._setup_custom_components()

    def _download_nltk_data(self):
        """Download required NLTK data with error handling."""
        nltk_resources = ['punkt', 'stopwords', 'averaged_perceptron_tagger']
        for resource in nltk_resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                except Exception as e:
                    self.logger.warning(f"Impossibile scaricare la risorsa NLTK {resource}: {str(e)}")

    def _setup_custom_components(self):
        """Configure custom spaCy pipeline components."""
        if not Language.has_factory("custom_sentencizer"):
            @Language.factory("custom_sentencizer")
            def create_custom_sentencizer(nlp, name):
                return CustomSentencizer()

        if "custom_sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("custom_sentencizer", before="parser")

    def process_text(self, text: str) -> Dict:
        """
        Process text and extract linguistic information.

        Args:
            text (str): Input text to process

        Returns:
            Dict: Dictionary containing extracted linguistic information
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")

        try:
            doc = self.nlp(text)
            return {
                'entities': [(ent.text, ent.label_) for ent in doc.ents],
                'tokens': [token.text for token in doc],
                'pos_tags': [(token.text, token.pos_) for token in doc],
                'sentences': [str(sent) for sent in doc.sents],
                'noun_chunks': [chunk.text for chunk in doc.noun_chunks],
                'dependencies': [(token.text, token.dep_, token.head.text) for token in doc],
                'lemmas': [(token.text, token.lemma_) for token in doc],
                'is_fallback': self.is_fallback,
                'model_info': {
                    'name': self.model_name,
                    'language': self.nlp.lang
                }
            }
        except Exception as e:
            self.logger.error(f"Errore nell'elaborazione del testo: {str(e)}")
            raise

    def analyze_sentiment(self, text: str) -> Optional[Dict]:
        """
        Perform comprehensive sentiment analysis.

        Args:
            text (str): Input text for sentiment analysis

        Returns:
            Optional[Dict]: Sentiment analysis results or None if analysis fails
        """
        if not self.sentiment_analyzer:
            self.logger.warning("Analizzatore sentimenti non disponibile")
            return None

        try:
            bert_result = self.sentiment_analyzer(text)[0]
            blob = TextBlob(text)

            return {
                'bert_sentiment': {
                    'label': bert_result['label'],
                    'score': bert_result['score'],
                    'confidence': self._calculate_confidence(bert_result['score'])
                },
                'textblob_sentiment': {
                    'polarity': blob.sentiment.polarity,
                    'subjectivity': blob.sentiment.subjectivity
                },
                'emotion_analysis': self._analyze_emotions(text),
                'aspect_based': self._analyze_aspect_based_sentiment(text)
            }
        except Exception as e:
            self.logger.error(f"Errore nell'analisi del sentimento: {str(e)}")
            return None

    def _calculate_confidence(self, score: float) -> float:
        """Calculate confidence score."""
        return min(max(abs(score - 0.5) * 2, 0), 1)

    def _analyze_emotions(self, text: str) -> Dict[str, float]:
        """
        Analyze emotions in text.

        Args:
            text (str): Input text for emotion analysis

        Returns:
            Dict[str, float]: Dictionary of emotion scores
        """
        # Espandi le keywords per ogni emozione
        emotion_keywords = {
            'joy': ['felice', 'contento', 'gioioso', 'entusiasta', 'allegro', 'sereno'],
            'anger': ['arrabbiato', 'furioso', 'irritato', 'indignato', 'adirato'],
            'sadness': ['triste', 'malinconico', 'depresso', 'deluso', 'sconfortato'],
            'fear': ['spaventato', 'terrorizzato', 'ansioso', 'preoccupato', 'inquieto'],
            'surprise': ['sorpreso', 'stupito', 'meravigliato', 'incredulo', 'sbalordito'],
            'trust': ['fiducioso', 'sicuro', 'affidabile', 'fedele'],
            'disgust': ['disgustato', 'nauseato', 'repulsione', 'ribrezzo'],
            'anticipation': ['attesa', 'speranza', 'aspettativa', 'previsione']
        }

        text_lower = text.lower()
        emotions = {}
        words = len(text.split())

        for emotion, keywords in emotion_keywords.items():
            count = sum(1 for word in keywords if word in text_lower)
            emotions[emotion] = count / words if words > 0 else 0

        return emotions

    def _analyze_aspect_based_sentiment(self, text: str) -> List[Dict]:
        """Perform aspect-based sentiment analysis."""
        try:
            doc = self.nlp(text)
            aspects = []

            for sent in doc.sents:
                for chunk in sent.noun_chunks:
                    # Analyze sentiment around the aspect
                    context_start = max(0, chunk.start - 5)
                    context_end = min(len(doc), chunk.end + 5)
                    context = doc[context_start:context_end].text

                    sentiment = TextBlob(context).sentiment

                    aspects.append({
                        'aspect': chunk.text,
                        'sentiment': sentiment.polarity,
                        'context': context,
                        'confidence': abs(sentiment.polarity) * (1 - sentiment.subjectivity)
                    })

            return aspects
        except Exception as e:
            self.logger.error(f"Errore nell'analisi del sentimento basata sugli aspetti: {str(e)}")
            return []

    def summarize_text(self, text: str, max_length: int = 130, min_length: int = 30) -> Optional[str]:
        """
        Generate a summary of the text.

        Args:
            text (str): Input text to summarize
            max_length (int): Maximum length of the summary
            min_length (int): Minimum length of the summary

        Returns:
            Optional[str]: Generated summary or None if summarization fails
        """
        if not self.summarizer:
            self.logger.warning("Summarizer non disponibile")
            return None

        try:
            summary = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            self.logger.error(f"Errore nella generazione del riassunto: {str(e)}")
            return None

    def analyze_readability(self, text: str) -> Dict:
        """
        Analyze text readability with multiple metrics.

        Args:
            text (str): Input text for readability analysis

        Returns:
            Dict: Dictionary containing readability metrics
        """
        try:
            sentences = sent_tokenize(text)
            words = word_tokenize(text)

            # Base statistics
            num_sentences = len(sentences)
            num_words = len(words)
            num_chars = len(text)
            num_syllables = self._count_syllables(text)

            # Calculate readability metrics
            avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0
            avg_word_length = num_chars / num_words if num_words > 0 else 0
            avg_syllables_per_word = num_syllables / num_words if num_words > 0 else 0

            # Calculate multiple readability indices
            gulpease = self._calculate_gulpease(num_chars, num_words, num_sentences)
            flesch = self._calculate_flesch(avg_sentence_length, avg_syllables_per_word)

            return {
                'statistics': {
                    'num_sentences': num_sentences,
                    'num_words': num_words,
                    'num_chars': num_chars,
                    'num_syllables': num_syllables,
                    'avg_sentence_length': round(avg_sentence_length, 2),
                    'avg_word_length': round(avg_word_length, 2),
                    'avg_syllables_per_word': round(avg_syllables_per_word, 2)
                },
                'readability_scores': {
                    'gulpease_index': round(gulpease, 2),
                    'flesch_index': round(flesch, 2),
                    'complexity_level': self._get_complexity_level(gulpease)
                },
                'text_structure': {
                    'paragraph_count': text.count('\n\n') + 1,
                    'sentence_types': self._analyze_sentence_types(sentences),
                    'vocabulary_richness': self._calculate_vocabulary_richness(words)
                }
            }
        except Exception as e:
            self.logger.error(f"Errore nell'analisi della leggibilità: {str(e)}")
            raise

    def _count_syllables(self, text: str) -> int:
        """Count syllables in text using a simple heuristic approach."""

        def count_syllables_in_word(word: str) -> int:
            word = word.lower()
            count = 0
            vowels = 'aeiouy'
            on_vowel = False

            for char in word:
                is_vowel = char in vowels
                if is_vowel and not on_vowel:
                    count += 1
                on_vowel = is_vowel

            # Adjust for 'e' at end of word
            if word.endswith('e'):
                count -= 1
            # If word has no vowels, set count to 1
            if count == 0:
                count = 1
            return count

        words = text.split()
        return sum(count_syllables_in_word(word) for word in words)

    def _calculate_gulpease(self, num_chars: int, num_words: int, num_sentences: int) -> float:
        """Calculate Gulpease readability index."""
        if num_words == 0:
            return 0
        return 89 + (300 * num_sentences - 10 * num_chars) / num_words

    def _calculate_flesch(self, avg_sentence_length: float, avg_syllables_per_word: float) -> float:
        """Calculate Flesch readability index."""
        return 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)

    def _calculate_vocabulary_richness(self, words: List[str]) -> Dict[str, float]:
        """Calculate vocabulary richness metrics."""
        if not words:
            return {
                'type_token_ratio': 0,
                'hapax_ratio': 0,
                'vocabulary_density': 0
            }

        # Convert to lowercase for better counting
        words = [word.lower() for word in words]

        # Count unique words and their frequencies
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        unique_words = len(word_freq)
        total_words = len(words)
        hapax_legomena = sum(1 for word, freq in word_freq.items() if freq == 1)

        return {
            'type_token_ratio': unique_words / total_words if total_words > 0 else 0,
            'hapax_ratio': hapax_legomena / total_words if total_words > 0 else 0,
            'vocabulary_density': unique_words / (total_words ** 0.5) if total_words > 0 else 0
        }

    def _get_complexity_level(self, gulpease_score: float) -> str:
        """Determine complexity level based on Gulpease index."""
        if gulpease_score >= 80:
            return "molto facile"
        elif gulpease_score >= 60:
            return "facile"
        elif gulpease_score >= 40:
            return "medio"
        else:
            return "difficile"

    def _analyze_sentence_types(self, sentences: List[str]) -> Dict[str, int]:
        """Analyze the types of sentences in the text."""
        types = {
            'dichiarative': 0,
            'interrogative': 0,
            'esclamative': 0,
            'imperative': 0
        }

        imperative_indicators = [
            'per favore', 'fai', 'vai', 'dimmi', 'guarda',
            'ascolta', 'pensa', 'prova', 'cerca', 'usa'
        ]

        for sentence in sentences:
            sentence_lower = sentence.lower()
            if '?' in sentence:
                types['interrogative'] += 1
            elif '!' in sentence:
                types['esclamative'] += 1
            elif any(indicator in sentence_lower for indicator in imperative_indicators):
                types['imperative'] += 1
            else:
                types['dichiarative'] += 1

        return types

    def detect_language(self, text: str) -> Dict:
        """
        Detect the language of the text with confidence.

        Args:
            text (str): Input text

        Returns:
            Dict: Dictionary containing language detection results
        """
        try:
            lang = detect(text)
            # Calculate approximate confidence
            confidence = len(re.findall(r'\b\w+\b', text)) / 100.0
            confidence = min(max(confidence, 0.1), 0.99)

            return {
                'language': lang,
                'confidence': confidence,
                'is_reliable': confidence > 0.5,
                'supported_model': lang in self.SUPPORTED_MODELS
            }
        except Exception as e:
            self.logger.error(f"Errore nel rilevamento della lingua: {str(e)}")
            raise

    def translate_text(self, text: str, target_lang: str = "en") -> Optional[str]:
        """
        Translate text to specified language.

        Args:
            text (str): Text to translate
            target_lang (str): Target language code

        Returns:
            Optional[str]: Translated text or None if translation fails
        """
        if not self.translator:
            self.logger.warning("Traduttore non disponibile")
            return None

        try:
            translation = self.translator(text, target_language=target_lang)
            return translation[0]['translation_text']
        except Exception as e:
            self.logger.error(f"Errore nella traduzione: {str(e)}")
            return None

    def extract_topics(self, text: str, num_topics: int = 5) -> Optional[List[Dict]]:
        """
        Extract main topics from text.

        Args:
            text (str): Input text
            num_topics (int): Number of topics to extract

        Returns:
            Optional[List[Dict]]: List of extracted topics or None if extraction fails
        """
        try:
            # Preprocessing
            doc = self.nlp(text)
            sentences = [sent.text for sent in doc.sents]

            # Calculate TF-IDF
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(sentences)

            # Extract most important terms for topics
            feature_names = vectorizer.get_feature_names_out()
            scores = np.mean(tfidf_matrix.toarray(), axis=0)
            top_indices = np.argsort(scores)[-num_topics:]

            topics = []
            for idx in reversed(top_indices):
                topics.append({
                    'topic': feature_names[idx],
                    'score': float(scores[idx]),
                    'related_terms': self._find_related_terms(feature_names[idx], text)
                })

            return topics
        except Exception as e:
            self.logger.error(f"Errore nell'estrazione dei topic: {str(e)}")
            return None

    def _find_related_terms(self, term: str, text: str, num_terms: int = 3) -> List[str]:
        """Find terms related to a keyword."""
        try:
            doc = self.nlp(text)
            term_token = self.nlp(term)[0]

            similarities = []
            for token in doc:
                if not token.is_stop and not token.is_punct and token.text.lower() != term.lower():
                    similarities.append((token.text, token.similarity(term_token)))

            return [word for word, _ in sorted(similarities, key=lambda x: x[1], reverse=True)[:num_terms]]
        except Exception as e:
            self.logger.warning(f"Errore nel trovare termini correlati: {str(e)}")
            return []

    def answer_question(self, context: str, question: str) -> Optional[Dict]:
        """
        Answer questions based on context.

        Args:
            context (str): Context text
            question (str): Question to answer

        Returns:
            Optional[Dict]: Answer information or None if answering fails
        """
        if not self.question_answerer:
            self.logger.warning("Question answerer non disponibile")
            return None

        try:
            result = self.question_answerer(question=question, context=context)
            return {
                'answer': result['answer'],
                'score': result['score'],
                'start': result['start'],
                'end': result['end'],
                'confidence': result['score']
            }
        except Exception as e:
            self.logger.error(f"Errore nella risposta alla domanda: {str(e)}")
            return None

    def extract_patterns(self, text: str) -> Dict:
        """
        Extract linguistic patterns from text.

        Args:
            text (str): Input text

        Returns:
            Dict: Dictionary containing extracted patterns
        """
        try:
            doc = self.nlp(text)
            patterns = {
                'named_entities': self._categorize_entities(doc),
                'phrasal_patterns': self._extract_phrasal_patterns(doc),
                'syntax_patterns': self._analyze_syntax_patterns(doc),
                'colocations': self._find_collocations(doc)
            }
            return patterns
        except Exception as e:
            self.logger.error(f"Errore nell'estrazione dei pattern: {str(e)}")
            raise

    def _find_collocations(self, doc: Doc) -> List[Dict]:
        """Find word collocations in text."""
        collocations = []
        for i in range(len(doc) - 1):
            if not doc[i].is_stop and not doc[i + 1].is_stop:
                if doc[i].pos_ in ['NOUN', 'ADJ'] and doc[i + 1].pos_ in ['NOUN', 'ADJ']:
                    collocations.append({
                        'words': (doc[i].text, doc[i + 1].text),
                        'pos': (doc[i].pos_, doc[i + 1].pos_),
                        'dep': (doc[i].dep_, doc[i + 1].dep_)
                    })
        return collocations