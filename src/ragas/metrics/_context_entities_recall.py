from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
from langchain.pydantic_v1 import BaseModel

from ragas.llms.output_parser import RagasoutputParser, get_json_format_instructions
from ragas.llms.prompt import Prompt
from ragas.metrics.base import EvaluationMode, MetricWithLLM

if t.TYPE_CHECKING:
    from langchain.callbacks.base import Callbacks

logger = logging.getLogger(__name__)


class ContextEntitiesResponse(BaseModel):
    entities: t.List[str]


_output_instructions = get_json_format_instructions(
    pydantic_object=ContextEntitiesResponse
)
_output_parser = RagasoutputParser(pydantic_object=ContextEntitiesResponse)


TEXT_ENTITY_EXTRACTION = Prompt(
    name="text_entity_extraction",
    # instruction="""Given a text, extract unique entities without repetition. Ensure you consider different forms or mentions of the same entity as a single entity.""",
    instruction="""给定一段文本，提取其中的唯一实体而不重复。确保你将同一实体的不同形式或提及视为单个实体。""",
    input_keys=["text"],
    output_key="output",
    output_type="json",
    output_format_instruction=_output_instructions,
    # examples=[
    #     {
    #         "text": """The Eiffel Tower, located in Paris, France, is one of the most iconic landmarks globally.
    #         Millions of visitors are attracted to it each year for its breathtaking views of the city.
    #         Completed in 1889, it was constructed in time for the 1889 World's Fair.""",
    #         "output": ContextEntitiesResponse.parse_obj(
    #             {
    #                 "entities": [
    #                     "Eiffel Tower",
    #                     "Paris",
    #                     "France",
    #                     "1889",
    #                     "World's Fair",
    #                 ],
    #             }
    #         ).dict(),
    #     },
    #     {
    #         "text": """The Colosseum in Rome, also known as the Flavian Amphitheatre, stands as a monument to Roman architectural and engineering achievement.
    #         Construction began under Emperor Vespasian in AD 70 and was completed by his son Titus in AD 80.
    #         It could hold between 50,000 and 80,000 spectators who watched gladiatorial contests and public spectacles.""",
    #         "output": ContextEntitiesResponse.parse_obj(
    #             {
    #                 "entities": [
    #                     "Colosseum",
    #                     "Rome",
    #                     "Flavian Amphitheatre",
    #                     "Vespasian",
    #                     "AD 70",
    #                     "Titus",
    #                     "AD 80",
    #                 ],
    #             }
    #         ).dict(),
    #     },
    #     {
    #         "text": """The Great Wall of China, stretching over 21,196 kilometers from east to west, is a marvel of ancient defensive architecture.
    #         Built to protect against invasions from the north, its construction started as early as the 7th century BC.
    #         Today, it is a UNESCO World Heritage Site and a major tourist attraction.""",
    #         "output": ContextEntitiesResponse.parse_obj(
    #             {
    #                 "entities": [
    #                     "Great Wall of China",
    #                     "21,196 kilometers",
    #                     "7th century BC",
    #                     "UNESCO World Heritage Site",
    #                 ],
    #             }
    #         ).dict(),
    #     },
    #     {
    #         "text": """The Apollo 11 mission, which launched on July 16, 1969, marked the first time humans landed on the Moon.
    #         Astronauts Neil Armstrong, Buzz Aldrin, and Michael Collins made history, with Armstrong being the first man to step on the lunar surface.
    #         This event was a significant milestone in space exploration.""",
    #         "output": ContextEntitiesResponse.parse_obj(
    #             {
    #                 "entities": [
    #                     "Apollo 11 mission",
    #                     "July 16, 1969",
    #                     "Moon",
    #                     "Neil Armstrong",
    #                     "Buzz Aldrin",
    #                     "Michael Collins",
    #                 ],
    #             }
    #         ).dict(),
    #     },
    # ],
    examples=[
        {
            "text": """埃菲尔铁塔位于法国巴黎，是全球最著名的地标之一。
            每年有数以百万计的游客被其城市的壮丽景色所吸引。
            埃菲尔铁塔于1889年完工，恰逢1889年世界博览会之际建成。""",
            "output": ContextEntitiesResponse.parse_obj(
                {
                "entities": [
                    "埃菲尔铁塔",
                    "巴黎",
                    "法国",
                    "1889",
                    "世界博览会",
                ],
            }).dict(),
        },
        {
            "text": """罗马斗兽场，又称为弗拉维圆形剧场，是罗马建筑与工程成就的纪念碑。
            在公元70年由韦帕芗皇帝开始建造，后由其子提图斯于公元80年完成。
            它可以容纳5万至8万名观众观看角斗士比赛和公众表演。""",
            "output": ContextEntitiesResponse.parse_obj(
            {
                "entities": [
                    "罗马斗兽场",
                    "罗马",
                    "弗拉维圆形剧场",
                    "韦帕芗",
                    "公元70年",
                    "提图斯",
                    "公元80年",
                ],
            }).dict(),
        },
        {
            "text": """中国的长城，从东到西绵延超过21,196公里，是古代防御建筑的奇迹。
            为了抵御北方的入侵，其建设最早可追溯至公元前7世纪。
            如今，它是一个联合国教科文组织的世界遗产地，也是一个主要的旅游景点。""",
            "output": ContextEntitiesResponse.parse_obj(
                {
                "entities": [
                    "中国的长城",
                    "21,196公里",
                    "公元前7世纪",
                    "联合国教科文组织的世界遗产地",
                ],
            }).dict(),
        },
        {
            "text": """阿波罗11号任务于1969年7月16日发射，标志着人类首次登月。
            宇航员尼尔·阿姆斯特朗、巴兹·奥尔德林和迈克尔·柯林斯创造了历史，阿姆斯特朗成为第一个踏上月球表面的人。
            这一事件是太空探索中的一个重要里程碑。""",
            "output": ContextEntitiesResponse.parse_obj(
                {
                "entities": [
                    "阿波罗11号任务",
                    "1969年7月16日",
                    "月球",
                    "尼尔·阿姆斯特朗",
                    "巴兹·奥尔德林",
                    "迈克尔·柯林斯",
                ],
            }).dict(),
        },
    ]
)


@dataclass
class ContextEntityRecall(MetricWithLLM):
    """
    Calculates recall based on entities present in ground truth and context.
    Let CN be the set of entities present in context,
    GN be the set of entities present in the ground truth.

    Then we define can the context entity recall as follows:
    Context Entity recall = | CN ∩ GN | / | GN |

    If this quantity is 1, we can say that the retrieval mechanism has
    retrieved context which covers all entities present in the ground truth,
    thus being a useful retrieval. Thus this can be used to evaluate retrieval
    mechanisms in specific use cases where entities matter, for example, a
    tourism help chatbot.

    Attributes
    ----------
    name : str
    batch_size : int
        Batch size for openai completion.
    """

    name: str = "context_entity_recall"  # type: ignore
    evaluation_mode: EvaluationMode = EvaluationMode.gc  # type: ignore
    context_entity_recall_prompt: Prompt = field(
        default_factory=lambda: TEXT_ENTITY_EXTRACTION
    )
    batch_size: int = 15
    max_retries: int = 1

    def _compute_score(
        self, ground_truth_entities: t.Sequence[str], context_entities: t.Sequence[str]
    ) -> float:
        num_entities_in_both = len(
            set(context_entities).intersection(set(ground_truth_entities))
        )
        return num_entities_in_both / (len(ground_truth_entities) + 1e-8)

    async def get_entities(
        self,
        text: str,
        callbacks: Callbacks,
        is_async: bool,
    ) -> t.Optional[ContextEntitiesResponse]:
        assert self.llm is not None, "LLM is not initialized"
        p_value = self.context_entity_recall_prompt.format(
            text=text,
        )
        result = await self.llm.generate(
            prompt=p_value,
            callbacks=callbacks,
            is_async=is_async,
        )

        result_text = result.generations[0][0].text
        answer = await _output_parser.aparse(
            result_text, p_value, self.llm, self.max_retries
        )
        if answer is None:
            return ContextEntitiesResponse(entities=[])

        return answer

    async def _ascore(
        self,
        row: Dict,
        callbacks: Callbacks,
        is_async: bool,
    ) -> float:
        ground_truth, contexts = row["ground_truth"], row["contexts"]
        ground_truth = await self.get_entities(
            ground_truth, callbacks=callbacks, is_async=is_async
        )
        contexts = await self.get_entities(
            "\n".join(contexts), callbacks=callbacks, is_async=is_async
        )
        if ground_truth is None or contexts is None:
            return np.nan
        return self._compute_score(ground_truth.entities, contexts.entities)

    def save(self, cache_dir: str | None = None) -> None:
        return self.context_entity_recall_prompt.save(cache_dir)


context_entity_recall = ContextEntityRecall(batch_size=15)
