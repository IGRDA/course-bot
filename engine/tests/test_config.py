"""Unit tests for workflows/config/base.py (CourseConfig)."""

from workflows.config import CourseConfig


class TestCourseConfigDefaults:
    def test_default_construction(self):
        config = CourseConfig()
        assert config.title == ""
        assert config.text_llm_provider == "mistral"
        assert config.total_pages == 50
        assert config.language == "English"
        assert config.target_audience is None

    def test_basic_fields(self):
        config = CourseConfig(title="Python 101", language="Spanish", total_pages=100)
        assert config.title == "Python 101"
        assert config.language == "Spanish"
        assert config.total_pages == 100

    def test_nested_configs_have_defaults(self):
        config = CourseConfig()
        assert config.research is not None
        assert config.activities is not None
        assert config.html is not None
        assert config.image is not None
        assert config.podcast is not None
        assert config.bibliography is not None
        assert config.video is not None
        assert config.people is not None
        assert config.mindmap is not None


class TestFlatToNestedBackwardCompat:
    def test_enable_research_flat(self):
        config = CourseConfig(enable_research=True)
        assert config.research.enabled is True
        assert config.enable_research is True

    def test_enable_research_false_flat(self):
        config = CourseConfig(enable_research=False)
        assert config.research.enabled is False

    def test_research_max_queries_flat(self):
        config = CourseConfig(research_max_queries=10)
        assert config.research.max_queries == 10
        assert config.research_max_queries == 10

    def test_html_concurrency_flat(self):
        config = CourseConfig(html_concurrency=4)
        assert config.html.concurrency == 4
        assert config.html_concurrency == 4

    def test_generate_images_flat(self):
        config = CourseConfig(generate_images=False)
        assert config.image.enabled is False
        assert config.generate_images is False

    def test_generate_bibliography_flat(self):
        config = CourseConfig(generate_bibliography=False)
        assert config.bibliography.enabled is False
        assert config.generate_bibliography is False

    def test_generate_videos_flat(self):
        config = CourseConfig(generate_videos=False)
        assert config.video.enabled is False
        assert config.generate_videos is False

    def test_generate_mindmap_flat(self):
        config = CourseConfig(generate_mindmap=False)
        assert config.mindmap.enabled is False
        assert config.generate_mindmap is False

    def test_books_per_module_flat(self):
        config = CourseConfig(bibliography_books_per_module=5)
        assert config.bibliography.books_per_module == 5
        assert config.bibliography_books_per_module == 5

    def test_video_search_provider_flat(self):
        config = CourseConfig(video_search_provider="youtube")
        assert config.video.search_provider == "youtube"
        assert config.video_search_provider == "youtube"

    def test_people_per_module_flat(self):
        config = CourseConfig(people_per_module=3)
        assert config.people.people_per_module == 3
        assert config.people_per_module == 3


class TestFlatWithExistingNestedObject:
    """Tests for the branch where flat keys are merged into an existing nested config object."""

    def test_flat_key_merges_into_existing_object(self):
        from workflows.config.research import ResearchConfig

        # Pass a ResearchConfig object AND a flat key that maps to research
        # This triggers the `else` branch (existing is not a dict)
        config = CourseConfig(research=ResearchConfig(max_queries=3), enable_research=False)
        assert config.research.enabled is False
        assert config.research.max_queries == 3

    def test_theory_only_with_existing_research_object(self):
        from workflows.config.research import ResearchConfig

        # theory_only=True when research is already a ResearchConfig object
        config = CourseConfig(theory_only=True, research=ResearchConfig(enabled=True))
        assert config.research.enabled is False

    def test_non_dict_data_passthrough(self):
        # The validator returns data as-is if not a dict.
        # Test by passing a pre-constructed CourseConfig to model_validate
        config = CourseConfig(title="Test")
        # model_validate on the dict representation works normally
        data = config.model_dump()
        config2 = CourseConfig.model_validate(data)
        assert config2.title == "Test"


class TestTheoryOnly:
    def test_theory_only_disables_research(self):
        config = CourseConfig(theory_only=True)
        assert config.research.enabled is False
        assert config.use_reflection is False

    def test_theory_only_false_leaves_research_enabled(self):
        config = CourseConfig(theory_only=False)
        # Default research state should be unchanged
        assert config.theory_only is False

    def test_theory_only_overrides_explicit_research_enabled(self):
        # Even if user explicitly sets enable_research=True, theory_only wins
        config = CourseConfig(theory_only=True, enable_research=True)
        assert config.research.enabled is False


class TestPropertyAliases:
    def test_max_retries_property(self):
        config = CourseConfig(max_retries=5)
        assert config.max_retries == 5

    def test_concurrency_property(self):
        config = CourseConfig(concurrency=16)
        assert config.concurrency == 16

    def test_web_search_provider(self):
        config = CourseConfig(web_search_provider="tavily")
        assert config.web_search_provider == "tavily"

    def test_podcast_target_words(self):
        config = CourseConfig(podcast_target_words=800)
        assert config.podcast_target_words == 800

    def test_mindmap_max_nodes(self):
        config = CourseConfig(mindmap_max_nodes=20)
        assert config.mindmap.max_nodes == 20
        assert config.mindmap_max_nodes == 20

    def test_html_select_mode_alias(self):
        config = CourseConfig(select_html="random")
        assert config.html.select_mode == "random"
        assert config.select_html == "random"

    def test_research_max_results_per_query(self):
        config = CourseConfig(research_max_results_per_query=8)
        assert config.research_max_results_per_query == 8

    def test_activities_concurrency(self):
        config = CourseConfig(activities_concurrency=4)
        assert config.activities_concurrency == 4

    def test_activity_selection_mode(self):
        config = CourseConfig(activity_selection_mode="deterministic")
        assert config.activity_selection_mode == "deterministic"

    def test_sections_per_activity(self):
        config = CourseConfig(sections_per_activity=2)
        assert config.sections_per_activity == 2

    def test_html_formats(self):
        config = CourseConfig(html_formats="paragraphs,accordion")
        assert config.html_formats == "paragraphs,accordion"

    def test_html_random_seed(self):
        config = CourseConfig(html_random_seed=42)
        assert config.html_random_seed == 42

    def test_include_quotes_in_html(self):
        config = CourseConfig(include_quotes_in_html=True)
        assert config.include_quotes_in_html is True

    def test_include_tables_in_html(self):
        config = CourseConfig(include_tables_in_html=False)
        assert config.include_tables_in_html is False

    def test_image_search_provider(self):
        config = CourseConfig(image_search_provider="bing")
        assert config.image_search_provider == "bing"

    def test_use_vision_ranking(self):
        config = CourseConfig(use_vision_ranking=True)
        assert config.use_vision_ranking is True

    def test_num_images_to_fetch(self):
        config = CourseConfig(num_images_to_fetch=5)
        assert config.num_images_to_fetch == 5

    def test_vision_llm_provider(self):
        config = CourseConfig(vision_llm_provider="pixtral")
        assert config.vision_llm_provider == "pixtral"

    def test_image_concurrency(self):
        config = CourseConfig(image_concurrency=4)
        assert config.image_concurrency == 4

    def test_imagetext2text_concurrency(self):
        config = CourseConfig(imagetext2text_concurrency=2)
        assert config.imagetext2text_concurrency == 2

    def test_vision_ranking_batch_size(self):
        config = CourseConfig(vision_ranking_batch_size=5)
        assert config.vision_ranking_batch_size == 5

    def test_podcast_tts_engine(self):
        config = CourseConfig(podcast_tts_engine="edge")
        assert config.podcast_tts_engine == "edge"

    def test_podcast_speaker_map(self):
        config = CourseConfig(podcast_speaker_map={"host": "v1", "guest": "v2"})
        assert config.podcast_speaker_map == {"host": "v1", "guest": "v2"}

    def test_bibliography_articles_per_module(self):
        config = CourseConfig(bibliography_articles_per_module=3)
        assert config.bibliography_articles_per_module == 3

    def test_book_search_provider(self):
        config = CourseConfig(book_search_provider="googlebooks")
        assert config.book_search_provider == "googlebooks"

    def test_article_search_provider(self):
        config = CourseConfig(article_search_provider="arxiv")
        assert config.article_search_provider == "arxiv"

    def test_videos_per_module(self):
        config = CourseConfig(videos_per_module=5)
        assert config.videos_per_module == 5

    def test_generate_people(self):
        config = CourseConfig(generate_people=False)
        assert config.generate_people is False

    def test_people_llm_provider(self):
        config = CourseConfig(people_llm_provider="openai")
        assert config.people_llm_provider == "openai"

    def test_people_concurrency(self):
        config = CourseConfig(people_concurrency=3)
        assert config.people_concurrency == 3

    def test_mindmap_llm_provider(self):
        config = CourseConfig(mindmap_llm_provider="gemini")
        assert config.mindmap_llm_provider == "gemini"
