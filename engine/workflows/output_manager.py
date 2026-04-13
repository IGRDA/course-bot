"""
Output manager for course generation workflow.
Handles creating run folders and saving step snapshots, modules, and final outputs.
"""

import os
from datetime import datetime

from agents.html_formatter.exporter import export_to_html
from workflows.state import CourseState


class OutputManager:
    """
    Manages output for a single workflow run.

    Creates a dedicated folder per run containing:
    - course.json: Full final course state
    - course.html: Final HTML export
    - module_0.json, module_1.json, ...: Individual module files
    - steps/: Subfolder with state snapshots after each workflow step
    """

    # Step names with their order numbers for consistent naming
    STEP_NAMES = {
        "index": 1,
        "theories": 2,
        "activities": 3,
        "metadata": 4,
        "html": 5,
        "images": 6,
        "videos": 7,
        "bibliography": 8,
        "people": 9,
        "mindmap": 10,
        "json2book": 11,
        "parse_markdown": 12,
        "inject_images": 13,
        "restructure": 14,
        "enrichments": 15,
        "validate_structure": 16,
    }

    def __init__(self, title: str, output_dir: str = "output"):
        """
        Initialize the output manager and create the run folder.

        Args:
            title: Course title (used to name the run folder)
            output_dir: Base output directory (default: "output")
        """
        self.output_dir = output_dir
        self.run_folder = self._create_run_folder(title)
        self.steps_folder = os.path.join(self.run_folder, "steps")
        os.makedirs(self.steps_folder, exist_ok=True)

    def _create_run_folder(self, title: str) -> str:
        """
        Create a timestamped run folder for this workflow execution.

        Args:
            title: Course title to include in folder name

        Returns:
            Path to the created run folder
        """
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Generate timestamped folder name (same format as before)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in title)
        safe_title = safe_title.replace(" ", "_")
        folder_name = f"{safe_title}_{timestamp}"

        run_folder = os.path.join(self.output_dir, folder_name)
        os.makedirs(run_folder, exist_ok=True)

        return run_folder

    def save_step(self, step_name: str, state: CourseState, update_canonical_outputs: bool = True) -> str:
        """
        Save a state snapshot after a workflow step.
        Also updates course.json, course.html, and module_*.json so partial runs
        produce the expected outputs even if a later step fails.

        Args:
            step_name: Name of the step (must be one of STEP_NAMES keys)
            state: The CourseState to save
            update_canonical_outputs: If True (default), also write course.json,
                course.html, and module_*.json from this state.

        Returns:
            Path to the saved step file
        """
        if step_name not in self.STEP_NAMES:
            raise ValueError(f"Unknown step name: {step_name}. Must be one of {list(self.STEP_NAMES.keys())}")

        step_num = self.STEP_NAMES[step_name]
        filename = f"{step_num:02d}_{step_name}.json"
        step_path = os.path.join(self.steps_folder, filename)

        with open(step_path, "w", encoding="utf-8") as f:
            f.write(state.model_dump_json(indent=2, by_alias=True))

        print(f"   💾 Step saved: steps/{filename}")
        if update_canonical_outputs:
            self.save_final(state)
            self.save_modules(state)
        return step_path

    def save_modules(self, state: CourseState) -> list[str]:
        """
        Extract and save individual module files.
        Same logic as stract_module.py - saves just the module data, not wrapped in CourseState.

        Args:
            state: The CourseState containing modules to extract

        Returns:
            List of paths to saved module files
        """
        module_paths = []

        for i, module in enumerate(state.modules):
            module_path = os.path.join(self.run_folder, f"module_{i}.json")
            with open(module_path, "w", encoding="utf-8") as f:
                # Save just the module dict, not wrapped in CourseState
                f.write(module.model_dump_json(indent=2, by_alias=True))
            module_paths.append(module_path)

        print(f"   💾 Modules saved: module_0.json ... module_{len(state.modules) - 1}.json")
        return module_paths

    def save_final(self, state: CourseState) -> tuple[str, str]:
        """
        Save the final course.json and course.html files.

        Args:
            state: The final CourseState to save

        Returns:
            Tuple of (json_path, html_path)
        """
        # Save JSON
        json_path = os.path.join(self.run_folder, "course.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(state.model_dump_json(indent=2, by_alias=True))

        # Save HTML
        html_path = os.path.join(self.run_folder, "course.html")
        export_to_html(state, html_path)

        print("   💾 Final output saved: course.json, course.html")
        return json_path, html_path

    def get_run_folder(self) -> str:
        """Get the path to the run folder."""
        return self.run_folder

    def get_step_path(self, step_name: str) -> str:
        """
        Get the expected path for a step file (whether it exists or not).

        Args:
            step_name: Name of the step

        Returns:
            Path where the step file would be saved
        """
        if step_name not in self.STEP_NAMES:
            raise ValueError(f"Unknown step name: {step_name}")

        step_num = self.STEP_NAMES[step_name]
        filename = f"{step_num:02d}_{step_name}.json"
        return os.path.join(self.steps_folder, filename)


def load_step(step_path: str) -> CourseState:
    """
    Load a CourseState from a step file.
    Useful for resuming a workflow from a specific step.

    Args:
        step_path: Path to the step JSON file

    Returns:
        The loaded CourseState
    """
    import json

    with open(step_path, encoding="utf-8") as f:
        data = json.load(f)

    return CourseState.model_validate(data)
