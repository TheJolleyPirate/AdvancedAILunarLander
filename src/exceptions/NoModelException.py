from src.novelty.NoveltyName import NoveltyName


class NoModelException(Exception):
    def __init__(self, novelty_name: str | NoveltyName):
        if isinstance(novelty_name, NoveltyName):
            novelty_name = novelty_name.value
        self.message = f"There is no pre-trained model for novelty: {novelty_name}"


