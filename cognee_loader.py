import cognee

import asyncio
from cognee.modules.search.types import SearchType

async def main():
    import os
    import pathlib

    # cognee knowledge graph will be created based on the text
    # and description of these files
    import os

    raw_text = os.path.join(
        os.path.abspath("."),
        "data",
        "raw_text.txt"
    )

    test_image = os.path.join(
        os.path.abspath("."),
        "data",
        "test_image.png"
    )

    user_profile = os.path.join(
        os.path.abspath("."),
        "data",
        "user_profile.yaml"
    )

    # Create a clean slate for cognee -- reset data and system state
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    # Add multimedia files and make them available for cognify
    await cognee.add([raw_text, test_image,user_profile ])

    # Create knowledge graph with cognee
    await cognee.cognify()
    import pathlib
    from cognee.api.v1.visualize import visualize_graph

    # Use the current working directory instead of __file__:
    notebook_dir = pathlib.Path.cwd()

    graph_file_path = (notebook_dir / ".artifacts" / "graph_visualization.html").resolve()

    # Make sure to convert to string if visualize_graph expects a string
    b = await visualize_graph(str(graph_file_path))

if __name__ == '__main__':
    asyncio.run(main())
