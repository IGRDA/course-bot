"""
CLI for testing bibliography generation iteratively.

Usage:
    # Test with a sample quantum theory module (Spanish)
    python -m agents.bibliography_generator --topic "Quantum Theory" --language es

    # Test with a sample machine learning module (English)
    python -m agents.bibliography_generator --topic "Machine Learning" --language en

    # Test with custom module title
    python -m agents.bibliography_generator --module-title "Wave-particle duality" --language en

    # Test with more books/articles
    python -m agents.bibliography_generator --topic "Quantum Theory" --num-books 5 --num-articles 5

    # Validate URLs
    python -m agents.bibliography_generator --topic "Quantum Theory" --validate-urls
"""

import argparse
import json
import logging
import sys
from typing import Any

from workflows.state import Module, Section, Submodule

from .agent import (
    _search_articles_for_module,
    _search_books_for_module,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# Sample test modules for different topics
SAMPLE_MODULES = {
    "quantum_es": Module(
        title="¿Por qué la materia se comporta como onda y partícula? Dualidad onda-partícula y el experimento de la doble rendija",
        id="1",
        index=1,
        description="Este módulo explora el comportamiento dual de la materia como onda y partícula, centrándose en la dualidad onda-partícula y el experimento de la doble rendija.",
        submodules=[
            Submodule(
                title="Fundamentos de la dualidad onda-partícula",
                index=1,
                description="Conceptos básicos de la dualidad onda-partícula",
                sections=[
                    Section(
                        title="Experimentos históricos que demostraron la dualidad",
                        index=1,
                        description="Experimentos clave como el de la doble rendija",
                        summary="Revisa los experimentos pioneros de Thomas Young y los estudios de Einstein sobre el efecto fotoeléctrico.",
                    ),
                    Section(
                        title="La ecuación de De Broglie",
                        index=2,
                        description="Relación entre longitud de onda y momento",
                        summary="Analiza la ecuación de De Broglie y su significado físico.",
                    ),
                ],
            )
        ],
    ),
    "quantum_en": Module(
        title="Wave-Particle Duality and the Double-Slit Experiment",
        id="1",
        index=1,
        description="This module explores the dual behavior of matter as both wave and particle, focusing on wave-particle duality and the double-slit experiment.",
        submodules=[
            Submodule(
                title="Foundations of Wave-Particle Duality",
                index=1,
                description="Basic concepts of wave-particle duality",
                sections=[
                    Section(
                        title="Historical experiments demonstrating duality",
                        index=1,
                        description="Key experiments like the double-slit experiment",
                        summary="Reviews Thomas Young's pioneering experiments and Einstein's photoelectric effect studies.",
                    ),
                    Section(
                        title="The De Broglie Equation",
                        index=2,
                        description="Relationship between wavelength and momentum",
                        summary="Analyzes the De Broglie equation and its physical meaning.",
                    ),
                ],
            )
        ],
    ),
    "ml_en": Module(
        title="Deep Learning and Neural Networks",
        id="1",
        index=1,
        description="This module covers deep learning fundamentals, neural network architectures, and their applications in modern AI systems.",
        submodules=[
            Submodule(
                title="Neural Network Fundamentals",
                index=1,
                description="Core concepts of neural networks",
                sections=[
                    Section(
                        title="Perceptrons and Multi-layer Networks",
                        index=1,
                        description="From single perceptrons to deep networks",
                        summary="Covers the evolution from simple perceptrons to complex multi-layer architectures.",
                    ),
                    Section(
                        title="Backpropagation and Gradient Descent",
                        index=2,
                        description="Training neural networks",
                        summary="Explains the backpropagation algorithm and optimization techniques.",
                    ),
                ],
            )
        ],
    ),
    "ml_es": Module(
        title="Aprendizaje Profundo y Redes Neuronales",
        id="1",
        index=1,
        description="Este módulo cubre los fundamentos del aprendizaje profundo, arquitecturas de redes neuronales y sus aplicaciones en sistemas de IA modernos.",
        submodules=[
            Submodule(
                title="Fundamentos de Redes Neuronales",
                index=1,
                description="Conceptos básicos de redes neuronales",
                sections=[
                    Section(
                        title="Perceptrones y Redes Multicapa",
                        index=1,
                        description="De perceptrones simples a redes profundas",
                        summary="Cubre la evolución de perceptrones simples a arquitecturas multicapa complejas.",
                    ),
                    Section(
                        title="Retropropagación y Descenso de Gradiente",
                        index=2,
                        description="Entrenamiento de redes neuronales",
                        summary="Explica el algoritmo de retropropagación y técnicas de optimización.",
                    ),
                ],
            )
        ],
    ),
    # ========================================================================
    # DIVERSE TEST TOPICS - For generalization testing
    # ========================================================================
    # English Topics (4 diverse domains)
    "renaissance_en": Module(
        title="Renaissance Art History: Masters and Movements",
        id="1",
        index=1,
        description="This module explores the artistic revolution of the Renaissance period, focusing on major artists, techniques, and cultural influences that shaped Western art history.",
        submodules=[
            Submodule(
                title="Italian Renaissance Masters",
                index=1,
                description="Study of Leonardo, Michelangelo, and Raphael",
                sections=[
                    Section(
                        title="Leonardo da Vinci: Art and Science",
                        index=1,
                        description="Leonardo's revolutionary techniques and scientific approach",
                        summary="Examines Leonardo's sfumato technique, anatomical studies, and masterworks like the Mona Lisa.",
                    ),
                    Section(
                        title="Michelangelo and the Human Form",
                        index=2,
                        description="Sculpture, painting, and architecture",
                        summary="Analyzes the Sistine Chapel ceiling and David sculpture.",
                    ),
                ],
            )
        ],
    ),
    "climate_en": Module(
        title="Climate Change and Ecosystem Dynamics",
        id="1",
        index=1,
        description="This module examines the impacts of climate change on global ecosystems, including biodiversity loss, habitat shifts, and conservation strategies.",
        submodules=[
            Submodule(
                title="Climate Science Fundamentals",
                index=1,
                description="Understanding climate systems and change drivers",
                sections=[
                    Section(
                        title="Greenhouse Effect and Global Warming",
                        index=1,
                        description="Physical mechanisms of climate change",
                        summary="Explains the greenhouse effect, carbon cycle, and anthropogenic contributions.",
                    ),
                    Section(
                        title="Ecosystem Responses to Climate Change",
                        index=2,
                        description="How ecosystems adapt and migrate",
                        summary="Covers species migration, phenological shifts, and ecosystem tipping points.",
                    ),
                ],
            )
        ],
    ),
    "cbt_en": Module(
        title="Cognitive Behavioral Therapy: Theory and Practice",
        id="1",
        index=1,
        description="This module covers the theoretical foundations and practical applications of Cognitive Behavioral Therapy for treating anxiety, depression, and other psychological disorders.",
        submodules=[
            Submodule(
                title="CBT Foundations",
                index=1,
                description="Core principles of cognitive behavioral therapy",
                sections=[
                    Section(
                        title="Cognitive Model of Psychopathology",
                        index=1,
                        description="How thoughts influence emotions and behavior",
                        summary="Introduces Beck's cognitive model and the concept of automatic negative thoughts.",
                    ),
                    Section(
                        title="Behavioral Techniques in CBT",
                        index=2,
                        description="Exposure therapy and behavioral activation",
                        summary="Covers systematic desensitization, behavioral experiments, and activity scheduling.",
                    ),
                ],
            )
        ],
    ),
    "roman_en": Module(
        title="Ancient Roman Architecture and Engineering",
        id="1",
        index=1,
        description="This module explores Roman architectural innovations including concrete construction, arches, aqueducts, and monumental buildings that influenced Western architecture.",
        submodules=[
            Submodule(
                title="Roman Engineering Innovations",
                index=1,
                description="Concrete, arches, and structural systems",
                sections=[
                    Section(
                        title="Roman Concrete and Construction",
                        index=1,
                        description="Revolutionary building materials and techniques",
                        summary="Examines opus caementicium, the Pantheon dome, and Roman construction methods.",
                    ),
                    Section(
                        title="Aqueducts and Infrastructure",
                        index=2,
                        description="Water systems and urban planning",
                        summary="Covers the Roman aqueduct system, sewers, and urban infrastructure.",
                    ),
                ],
            )
        ],
    ),
    # Spanish Topics (4 diverse domains)
    "siglo_oro_es": Module(
        title="Literatura del Siglo de Oro Español",
        id="1",
        index=1,
        description="Este módulo explora la literatura española del Siglo de Oro, incluyendo obras de Cervantes, Lope de Vega, Quevedo y Góngora, y su impacto en la cultura hispana.",
        submodules=[
            Submodule(
                title="Prosa y Narrativa del Siglo de Oro",
                index=1,
                description="El Quijote y la novela picaresca",
                sections=[
                    Section(
                        title="Don Quijote de la Mancha",
                        index=1,
                        description="La obra maestra de Cervantes",
                        summary="Analiza la estructura, temas y significado del Quijote como primera novela moderna.",
                    ),
                    Section(
                        title="La Novela Picaresca",
                        index=2,
                        description="Lazarillo de Tormes y el Buscón",
                        summary="Examina el género picaresco y su crítica social.",
                    ),
                ],
            )
        ],
    ),
    "economia_es": Module(
        title="Economía del Desarrollo y Políticas Públicas",
        id="1",
        index=1,
        description="Este módulo examina las teorías del desarrollo económico, la reducción de la pobreza, y las políticas públicas para promover el crecimiento en países en desarrollo.",
        submodules=[
            Submodule(
                title="Teorías del Desarrollo Económico",
                index=1,
                description="Enfoques clásicos y contemporáneos",
                sections=[
                    Section(
                        title="Modelos de Crecimiento Económico",
                        index=1,
                        description="Desde Solow hasta el crecimiento endógeno",
                        summary="Cubre los modelos neoclásicos y teorías del crecimiento endógeno.",
                    ),
                    Section(
                        title="Reducción de la Pobreza",
                        index=2,
                        description="Estrategias y programas de desarrollo",
                        summary="Analiza programas de transferencias condicionadas y microfinanzas.",
                    ),
                ],
            )
        ],
    ),
    "biologia_es": Module(
        title="Biología Marina y Ecosistemas Oceánicos",
        id="1",
        index=1,
        description="Este módulo estudia la diversidad de la vida marina, los ecosistemas oceánicos, y los impactos del cambio climático y la contaminación en los océanos.",
        submodules=[
            Submodule(
                title="Ecosistemas Marinos",
                index=1,
                description="Arrecifes, océano profundo y zonas costeras",
                sections=[
                    Section(
                        title="Arrecifes de Coral",
                        index=1,
                        description="Biodiversidad y amenazas",
                        summary="Examina la ecología de arrecifes, blanqueamiento de coral y conservación.",
                    ),
                    Section(
                        title="Cadenas Tróficas Marinas",
                        index=2,
                        description="Productividad primaria y redes alimentarias",
                        summary="Cubre el fitoplancton, zooplancton y flujos de energía oceánicos.",
                    ),
                ],
            )
        ],
    ),
    "derecho_es": Module(
        title="Derecho Constitucional y Derechos Fundamentales",
        id="1",
        index=1,
        description="Este módulo analiza los principios del derecho constitucional, la separación de poderes, y la protección de los derechos fundamentales en sistemas democráticos.",
        submodules=[
            Submodule(
                title="Fundamentos del Derecho Constitucional",
                index=1,
                description="Constitución, Estado de Derecho y democracia",
                sections=[
                    Section(
                        title="Separación de Poderes",
                        index=1,
                        description="Ejecutivo, legislativo y judicial",
                        summary="Analiza el sistema de checks and balances y la división del poder estatal.",
                    ),
                    Section(
                        title="Derechos Fundamentales",
                        index=2,
                        description="Garantías constitucionales y su protección",
                        summary="Examina los derechos civiles, políticos y sociales en la constitución.",
                    ),
                ],
            )
        ],
    ),
}


def validate_url(url: str, timeout: int = 5) -> tuple[bool, str]:
    """
    Validate a URL by making a HEAD request.

    Args:
        url: URL to validate
        timeout: Request timeout in seconds

    Returns:
        Tuple of (is_valid, status_message)
    """
    import requests

    if not url:
        return False, "Empty URL"

    try:
        # Use HEAD request to avoid downloading content
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        if response.status_code < 400:
            return True, f"OK ({response.status_code})"
        else:
            return False, f"HTTP {response.status_code}"
    except requests.exceptions.Timeout:
        return False, "Timeout"
    except requests.exceptions.ConnectionError:
        return False, "Connection error"
    except Exception as e:
        return False, str(e)[:50]


def print_book_result(book: Any, index: int, validate: bool = False) -> dict:
    """Print a single book result and return quality metrics."""
    print(f"\n  📚 {index}. {book.title}")
    print(f"      Authors: {', '.join(book.authors[:3])}")
    print(f"      Year: {book.year or 'N/A'}")
    print(f"      Publisher: {book.publisher or 'N/A'}")
    print(f"      ISBN: {book.isbn or book.isbn_13 or 'N/A'}")
    print(f"      URL: {book.url or 'N/A'}")

    metrics = {
        "has_authors": len(book.authors) > 0,
        "has_year": book.year is not None,
        "has_publisher": book.publisher is not None,
        "has_isbn": book.isbn is not None or book.isbn_13 is not None,
        "has_url": bool(book.url),
        "url_valid": None,
    }

    if validate and book.url:
        is_valid, status = validate_url(book.url)
        metrics["url_valid"] = is_valid
        status_icon = "✅" if is_valid else "❌"
        print(f"      URL Status: {status_icon} {status}")

    if book.apa_citation:
        print(f"      APA: {book.apa_citation[:100]}...")

    return metrics


def print_article_result(article: dict, index: int, validate: bool = False) -> dict:
    """Print a single article result and return quality metrics."""
    print(f"\n  📄 {index}. {article['title'][:80]}...")
    print(f"      Authors: {', '.join(article['authors'][:3])}")
    print(f"      Year: {article.get('year') or 'N/A'}")
    print(f"      Venue: {article.get('venue') or 'N/A'}")
    print(f"      Citations: {article.get('citation_count') or 'N/A'}")
    print(f"      Source: {article.get('source')}")
    print(f"      URL: {article.get('url', 'N/A')}")

    metrics = {
        "has_authors": len(article.get("authors", [])) > 0,
        "has_year": article.get("year") is not None,
        "has_venue": bool(article.get("venue")),
        "citation_count": article.get("citation_count") or 0,
        "has_abstract": bool(article.get("abstract")),
        "has_url": bool(article.get("url")),
        "url_valid": None,
        "source": article.get("source"),
    }

    if validate and article.get("url"):
        is_valid, status = validate_url(article["url"])
        metrics["url_valid"] = is_valid
        status_icon = "✅" if is_valid else "❌"
        print(f"      URL Status: {status_icon} {status}")

    if article.get("snippet"):
        print(f"      Snippet: {article['snippet'][:100]}...")

    return metrics


def calculate_quality_score(book_metrics: list, article_metrics: list) -> dict:
    """Calculate overall quality scores."""
    scores = {
        "books": {
            "total": len(book_metrics),
            "with_isbn": sum(1 for m in book_metrics if m["has_isbn"]),
            "with_year": sum(1 for m in book_metrics if m["has_year"]),
            "with_publisher": sum(1 for m in book_metrics if m["has_publisher"]),
            "valid_urls": sum(1 for m in book_metrics if m["url_valid"] is True),
            "invalid_urls": sum(1 for m in book_metrics if m["url_valid"] is False),
        },
        "articles": {
            "total": len(article_metrics),
            "with_venue": sum(1 for m in article_metrics if m["has_venue"]),
            "with_citations": sum(1 for m in article_metrics if m["citation_count"] > 0),
            "high_citations": sum(1 for m in article_metrics if m["citation_count"] >= 10),
            "with_abstract": sum(1 for m in article_metrics if m["has_abstract"]),
            "valid_urls": sum(1 for m in article_metrics if m["url_valid"] is True),
            "invalid_urls": sum(1 for m in article_metrics if m["url_valid"] is False),
            "avg_citations": (
                sum(m["citation_count"] for m in article_metrics) / len(article_metrics) if article_metrics else 0
            ),
        },
    }

    # Calculate overall score (0-100)
    book_score = 0
    if scores["books"]["total"] > 0:
        book_score = (
            (scores["books"]["with_isbn"] / scores["books"]["total"]) * 30
            + (scores["books"]["with_year"] / scores["books"]["total"]) * 20
            + (scores["books"]["with_publisher"] / scores["books"]["total"]) * 20
            + (min(scores["books"]["total"], 5) / 5) * 30  # Coverage score
        )

    article_score = 0
    if scores["articles"]["total"] > 0:
        article_score = (
            (scores["articles"]["with_venue"] / scores["articles"]["total"]) * 20
            + (scores["articles"]["high_citations"] / scores["articles"]["total"]) * 30
            + (scores["articles"]["with_abstract"] / scores["articles"]["total"]) * 20
            + (min(scores["articles"]["total"], 5) / 5) * 30  # Coverage score
        )

    scores["overall_score"] = (book_score + article_score) / 2

    return scores


def run_test(
    module: Module,
    course_title: str,
    language: str,
    num_books: int,
    num_articles: int,
    llm_provider: str,
    article_provider: str,
    validate_urls: bool,
) -> dict:
    """Run a single bibliography test."""

    print(f"\n{'=' * 70}")
    print("📖 Testing Bibliography Generation")
    print(f"{'=' * 70}")
    print(f"Course Title: {course_title}")
    print(f"Module: {module.title}")
    print(f"Language: {language}")
    print(f"Target: {num_books} books, {num_articles} articles")
    print(f"LLM Provider: {llm_provider}")
    print(f"Article Provider: {article_provider}")
    print(f"{'=' * 70}")

    existing_keys: set[str] = set()

    # Search for books
    print("\n🔍 Searching for books...")
    books, existing_keys = _search_books_for_module(
        module=module,
        course_title=course_title,
        language=language,
        provider=llm_provider,
        num_books=num_books,
        existing_keys=existing_keys,
    )

    print(f"\n📚 BOOKS FOUND: {len(books)}")
    book_metrics = []
    for i, book in enumerate(books, 1):
        metrics = print_book_result(book, i, validate=validate_urls)
        book_metrics.append(metrics)

    # Search for articles
    print("\n🔍 Searching for articles...")
    articles, existing_keys = _search_articles_for_module(
        module=module,
        language=language,
        article_provider=article_provider,
        num_articles=num_articles,
        existing_keys=existing_keys,
    )

    print(f"\n📄 ARTICLES FOUND: {len(articles)}")
    article_metrics = []
    for i, article in enumerate(articles, 1):
        metrics = print_article_result(article, i, validate=validate_urls)
        article_metrics.append(metrics)

    # Calculate quality scores
    scores = calculate_quality_score(book_metrics, article_metrics)

    print(f"\n{'=' * 70}")
    print("📊 QUALITY REPORT")
    print(f"{'=' * 70}")
    print(f"Books: {scores['books']['total']} found")
    print(f"  - With ISBN: {scores['books']['with_isbn']}")
    print(f"  - With Year: {scores['books']['with_year']}")
    print(f"  - With Publisher: {scores['books']['with_publisher']}")
    if validate_urls:
        print(f"  - Valid URLs: {scores['books']['valid_urls']}")
        print(f"  - Invalid URLs: {scores['books']['invalid_urls']}")

    print(f"\nArticles: {scores['articles']['total']} found")
    print(f"  - With Venue: {scores['articles']['with_venue']}")
    print(f"  - With Citations: {scores['articles']['with_citations']}")
    print(f"  - High Citations (10+): {scores['articles']['high_citations']}")
    print(f"  - Avg Citations: {scores['articles']['avg_citations']:.1f}")
    print(f"  - With Abstract: {scores['articles']['with_abstract']}")
    if validate_urls:
        print(f"  - Valid URLs: {scores['articles']['valid_urls']}")
        print(f"  - Invalid URLs: {scores['articles']['invalid_urls']}")

    print(f"\n⭐ OVERALL QUALITY SCORE: {scores['overall_score']:.1f}/100")
    print(f"{'=' * 70}")

    return {
        "books": [b.model_dump() if hasattr(b, "model_dump") else b for b in books],
        "articles": articles,
        "scores": scores,
    }


def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Test bibliography generation for course modules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m agents.bibliography_generator --topic quantum --language es
    python -m agents.bibliography_generator --topic ml --language en
    python -m agents.bibliography_generator --module-title "Neural Networks" --num-books 5
    python -m agents.bibliography_generator --topic quantum --validate-urls
        """,
    )

    parser.add_argument(
        "--topic",
        "-t",
        choices=[
            "quantum",
            "ml",  # Original STEM topics
            "renaissance",
            "climate",
            "cbt",
            "roman",  # English diverse topics
            "siglo_oro",
            "economia",
            "biologia",
            "derecho",  # Spanish diverse topics
        ],
        help="Use a predefined sample topic (quantum, ml, renaissance, climate, cbt, roman, siglo_oro, economia, biologia, derecho)",
    )

    parser.add_argument(
        "--module-title",
        "-m",
        help="Custom module title to test with",
    )

    parser.add_argument(
        "--language",
        "-l",
        choices=["en", "es"],
        default="en",
        help="Language for the content (default: en)",
    )

    parser.add_argument(
        "--num-books",
        "-b",
        type=int,
        default=5,
        help="Number of books to find (default: 5)",
    )

    parser.add_argument(
        "--num-articles",
        "-a",
        type=int,
        default=5,
        help="Number of articles to find (default: 5)",
    )

    parser.add_argument(
        "--llm-provider",
        default="mistral",
        help="LLM provider for book suggestions (default: mistral)",
    )

    parser.add_argument(
        "--article-provider",
        default="openalex",
        help="Article search provider (default: openalex)",
    )

    parser.add_argument(
        "--validate-urls",
        "-v",
        action="store_true",
        help="Validate URLs with HTTP HEAD requests",
    )

    parser.add_argument(
        "--output",
        "-o",
        help="Output results to JSON file",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine which module to use
    if args.topic:
        # Map topic to module key and course title
        topic_config = {
            "quantum": ("quantum", "Quantum Theory"),
            "ml": ("ml", "Machine Learning"),
            "renaissance": ("renaissance", "Renaissance Art History"),
            "climate": ("climate", "Climate Change"),
            "cbt": ("cbt", "Cognitive Behavioral Therapy"),
            "roman": ("roman", "Ancient Roman Architecture"),
            "siglo_oro": ("siglo_oro", "Literatura del Siglo de Oro"),
            "economia": ("economia", "Economía del Desarrollo"),
            "biologia": ("biologia", "Biología Marina"),
            "derecho": ("derecho", "Derecho Constitucional"),
        }

        topic_key, course_title = topic_config.get(args.topic, (args.topic, args.topic))

        # Build module key with language suffix
        key = f"{topic_key}_{args.language}"
        if key not in SAMPLE_MODULES:
            # Fallback to English version if Spanish not available
            key = f"{topic_key}_en"
            if key not in SAMPLE_MODULES:
                # Try Spanish version
                key = f"{topic_key}_es"

        if key not in SAMPLE_MODULES:
            parser.error(f"Topic '{args.topic}' with language '{args.language}' not found")
            return 1

        module = SAMPLE_MODULES[key]
    elif args.module_title:
        # Create a simple module from the title
        module = Module(
            title=args.module_title,
            id="1",
            index=1,
            description=f"Module about {args.module_title}",
            submodules=[
                Submodule(
                    title=args.module_title,
                    index=1,
                    description="",
                    sections=[
                        Section(
                            title=args.module_title,
                            index=1,
                            description="",
                            summary="",
                        )
                    ],
                )
            ],
        )
        course_title = args.module_title
    else:
        parser.error("Either --topic or --module-title is required")
        return 1

    language = "Español" if args.language == "es" else "English"

    # Run the test
    results = run_test(
        module=module,
        course_title=course_title,
        language=language,
        num_books=args.num_books,
        num_articles=args.num_articles,
        llm_provider=args.llm_provider,
        article_provider=args.article_provider,
        validate_urls=args.validate_urls,
    )

    # Save output if requested
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to {args.output}")

    # Return success if we found at least some results
    if results["scores"]["overall_score"] >= 50:
        return 0
    else:
        print("\n⚠️  Quality score below 50, consider improving search parameters")
        return 1


if __name__ == "__main__":
    sys.exit(main())
