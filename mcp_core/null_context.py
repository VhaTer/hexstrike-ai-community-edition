"""NullContext — faux contexte MCP pour les appels internes à run_security_tool().

Remplace le vrai Context FastMCP quand scan(), CTF engine ou bugbounty engine
appellent run_security_tool() sans passer par le protocole MCP.
Chaque méthode est no-op ou retourne un mock cohérent avec ce que
run_security_tool() attend vraiment (vérifié sur le code source réel).
"""

from typing import Any, Optional


class _DummyContent:
    """Simule TextResourceContents — un seul champ .content texte."""
    def __init__(self, text: str = ""):
        self.content = text


class _DummyResource:
    """Simule ReadResourceResult — .contents est une liste de _DummyContent."""
    def __init__(self):
        self.contents = []


class _DummySample:
    """Simule SampleResult — .text est la suggestion brute."""
    def __init__(self):
        self.text = ""


class NullContext:
    """Contexte MCP minimaliste no-op pour les appels internes.

    Toutes les méthodes async sont awaitables et ne lèvent jamais.
    Les ressources et suggestions retournent des objets vides cohérents
    (jamais None là où le code appelant attend .text ou .content).
    """

    session_id: str = ""

    async def info(self, message: str) -> None:
        pass

    async def warning(self, message: str) -> None:
        pass

    async def error(self, message: str) -> None:
        pass

    async def report_progress(self, current: int, total: int) -> None:
        pass

    async def read_resource(self, uri: str) -> _DummyResource:
        return _DummyResource()

    async def sample(self, messages: Any, **kwargs: Any) -> _DummySample:
        return _DummySample()

    async def set_state(self, key: str, value: Any) -> None:
        pass

    async def get_state(self, key: str) -> Any:
        return {}

    @property
    def request_id(self) -> Optional[str]:
        return None
