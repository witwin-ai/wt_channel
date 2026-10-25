"""A request to see a place's map always imports it into a new Scene first.

"Show me the UCSD map and run a radio map" while the UCSD Scene was on screen only
re-ran the radio map: the description said to import a map only when the Scene had
none, so the Agent reused the existing map instead of creating a new Scene.
"""
from wt_channel import agent_tools


def test_description_imports_a_requested_map_even_if_one_is_open():
    text = agent_tools._DESCRIPTION.lower()

    assert "if the scene has no map" not in text
    assert "even if a map of that place is already open" in text
    assert "run the radio map again" in text
