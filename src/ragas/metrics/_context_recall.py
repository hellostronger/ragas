from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from langchain_core.pydantic_v1 import BaseModel

from ragas.llms.output_parser import RagasoutputParser, get_json_format_instructions
from ragas.llms.prompt import Prompt
from ragas.metrics.base import EvaluationMode, MetricWithLLM, ensembler

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

    from ragas.llms.prompt import PromptValue

logger = logging.getLogger(__name__)


class ContextRecallClassificationAnswer(BaseModel):
    statement: str
    attributed: int
    reason: str


class ContextRecallClassificationAnswers(BaseModel):
    __root__: t.List[ContextRecallClassificationAnswer]

    def dicts(self) -> t.List[t.Dict]:
        return self.dict()["__root__"]


_classification_output_instructions = get_json_format_instructions(
    ContextRecallClassificationAnswers
)
_output_parser = RagasoutputParser(pydantic_object=ContextRecallClassificationAnswers)


CONTEXT_RECALL_RA = Prompt(
    name="context_recall",
    # instruction="""Given a context, and an answer, analyze each sentence in the answer and classify if the sentence can be attributed to the given context or not. Use only "Yes" (1) or "No" (0) as a binary classification. Output json with reason.""",
    instruction="""给定一个上下文和一个答案，分析答案中的每一句话，并判断这些句子是否可以归因于给定的上下文。仅使用'是'（1）或'否'（0）作为二元分类。输出时需附带理由，使用json格式。""",
    output_format_instruction=_classification_output_instructions,
    # examples=[
    #     {
    #         "question": """What can you tell me about albert Albert Einstein?""",
    #         "context": """Albert Einstein (14 March 1879 - 18 April 1955) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century. His mass-energy equivalence formula E = mc2, which arises from relativity theory, has been called 'the world's most famous equation'. He received the 1921 Nobel Prize in Physics 'for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect', a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time. His intellectual achievements and originality have made Einstein synonymous with genius.""",
    #         "answer": """Albert Einstein born in 14 March 1879 was  German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics for his services to theoretical physics. He published 4 papers in 1905.  Einstein moved to Switzerland in 1895""",
    #         "classification": ContextRecallClassificationAnswers.parse_obj(
    #             [
    #                 {
    #                     "statement": "Albert Einstein, born on 14 March 1879, was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time.",
    #                     "reason": "The date of birth of Einstein is mentioned clearly in the context.",
    #                     "attributed": 1,
    #                 },
    #                 {
    #                     "statement": "He received the 1921 Nobel Prize in Physics for his services to theoretical physics.",
    #                     "reason": "The exact sentence is present in the given context.",
    #                     "attributed": 1,
    #                 },
    #                 {
    #                     "statement": "He published 4 papers in 1905.",
    #                     "reason": "There is no mention about papers he wrote in the given context.",
    #                     "attributed": 0,
    #                 },
    #                 {
    #                     "statement": "Einstein moved to Switzerland in 1895.",
    #                     "reason": "There is no supporting evidence for this in the given context.",
    #                     "attributed": 0,
    #                 },
    #             ]
    #         ).dicts(),
    #     },
    #     {
    #         "question": """who won 2020 icc world cup?""",
    #         "context": """The 2022 ICC Men's T20 World Cup, held from October 16 to November 13, 2022, in Australia, was the eighth edition of the tournament. Originally scheduled for 2020, it was postponed due to the COVID-19 pandemic. England emerged victorious, defeating Pakistan by five wickets in the final to clinch their second ICC Men's T20 World Cup title.""",
    #         "answer": """England""",
    #         "classification": ContextRecallClassificationAnswers.parse_obj(
    #             [
    #                 {
    #                     "statement": "England won the 2022 ICC Men's T20 World Cup.",
    #                     "reason": "From context it is clear that England defeated Pakistan to win the World Cup.",
    #                     "attributed": 1,
    #                 },
    #             ]
    #         ).dicts(),
    #     },
    #     {
    #         "question": """What is the primary fuel for the Sun?""",
    #         "context": """NULL""",
    #         "answer": """Hydrogen""",
    #         "classification": ContextRecallClassificationAnswers.parse_obj(
    #             [
    #                 {
    #                     "statement": "The Sun's primary fuel is hydrogen.",
    #                     "reason": "The context contains no information",
    #                     "attributed": 0,
    #                 },
    #             ]
    #         ).dicts(),
    #     },
    # ],
examples=[
    {
        "question": """你能告诉我关于阿尔伯特·爱因斯坦的什么信息？""",
        "context": """阿尔伯特·爱因斯坦（1879年3月14日－1955年4月18日）是一位出生于德国的理论物理学家，被广泛认为是有史以来最伟大和最有影响力的科学家之一。他最出名的是发展了相对论，同时也对量子力学做出了重要贡献，因此成为了二十世纪初现代物理学革命性重塑自然科学理解的核心人物。他的质能等价公式E=mc²，源于相对论理论，被称为“世界上最著名的方程”。他因“他对理论物理学的贡献，特别是他发现了光电效应定律”而获得了1921年的诺贝尔物理学奖，这是量子理论发展中的一个关键步骤。他的工作也因其对科学哲学的影响而闻名。在1999年由英国《物理世界》杂志对全球130位顶尖物理学家进行的一项民意调查中，爱因斯坦被评为有史以来最伟大的物理学家。他的智慧成就和原创性使爱因斯坦与天才同义。""",
        "answer": """阿尔伯特·爱因斯坦生于1879年3月14日，是一位出生于德国的理论物理学家，被广泛认为是有史以来最伟大和最有影响力的科学家之一。他因对理论物理学的服务获得了1921年的诺贝尔物理学奖。他在1905年发表了4篇论文。爱因斯坦在1895年移居瑞士。""",
        "classification": ContextRecallClassificationAnswers.parse_obj(
            [
            {
                "statement": "阿尔伯特·爱因斯坦，生于1879年3月14日，是一位出生于德国的理论物理学家，被广泛认为是有史以来最伟大和最有影响力的科学家之一。",
                "reason": "爱因斯坦的出生日期在上下文中明确提及。",
                "attributed": 1,
            },
            {
                "statement": "他因对理论物理学的服务获得了1921年的诺贝尔物理学奖。",
                "reason": "确切的句子出现在给定的上下文中。",
                "attributed": 1,
            },
            {
                "statement": "他在1905年发表了4篇论文。",
                "reason": "在给定的上下文中没有提及他撰写的论文。",
                "attributed": 0,
            },
            {
                "statement": "爱因斯坦在1895年移居瑞士。",
                "reason": "在给定的上下文中没有支持这一说法的证据。",
                "attributed": 0,
            },
        ]
        ).dicts(),
    },
    {
        "question": """谁赢得了2020年国际板球理事会（ICC）世界杯？""",
        "context": """2022年ICC男子T20世界杯，从2022年10月16日至11月13日在澳大利亚举行，是这项赛事的第八届。最初定于2020年举行，但由于COVID-19大流行而推迟。英格兰在决赛中以五票之差击败巴基斯坦，赢得了自己的第二个ICC男子T20世界杯冠军。""",
        "answer": """英格兰""",
        "classification": ContextRecallClassificationAnswers.parse_obj(
            [
            {
                "statement": "英格兰赢得了2022年ICC男子T20世界杯。",
                "reason": "从上下文中可以看出，英格兰在决赛中击败巴基斯坦赢得了世界杯。",
                "attributed": 1,
            },
        ]
        ).dicts(),
    },
    {
        "question": """太阳的主要燃料是什么？""",
        "context": """NULL""",
        "answer": """氢气""",
        "classification": ContextRecallClassificationAnswers.parse_obj(
            [
            {
                "statement": "太阳的主要燃料是氢气。",
                "reason": "上下文中没有信息。",
                "attributed": 0,
            },
        ]
        ).dicts(),
    },
],
    input_keys=["question", "context", "answer"],
    output_key="classification",
    output_type="json",
)


@dataclass
class ContextRecall(MetricWithLLM):

    """
    Estimates context recall by estimating TP and FN using annotated answer and
    retrieved context.

    Attributes
    ----------
    name : str
    """

    name: str = "context_recall"  # type: ignore
    evaluation_mode: EvaluationMode = EvaluationMode.qcg  # type: ignore
    context_recall_prompt: Prompt = field(default_factory=lambda: CONTEXT_RECALL_RA)
    max_retries: int = 1
    _reproducibility: int = 1

    @property
    def reproducibility(self):
        return self._reproducibility

    @reproducibility.setter
    def reproducibility(self, value):
        if value < 1:
            logger.warning("reproducibility cannot be less than 1, setting to 1")
            value = 1
        elif value % 2 == 0:
            logger.warning(
                "reproducibility level cannot be set to even number, setting to odd"
            )
            value += 1
        self._reproducibility = value

    def __post_init__(self) -> None:
        if self.reproducibility < 1:
            logger.warning("reproducibility cannot be less than 1, setting to 1")
            self.reproducibility = 1

    def _create_context_recall_prompt(self, row: t.Dict) -> PromptValue:
        qstn, ctx, gt = row["question"], row["contexts"], row["ground_truth"]
        ctx = "\n".join(ctx) if isinstance(ctx, list) else ctx

        return self.context_recall_prompt.format(question=qstn, context=ctx, answer=gt)

    def _compute_score(self, response: t.Any) -> float:
        response = [1 if item.attributed else 0 for item in response.__root__]
        denom = len(response)
        numerator = sum(response)
        score = numerator / denom if denom > 0 else np.nan

        if np.isnan(score):
            logger.warning("The LLM did not return a valid classification.")

        return score

    async def _ascore(self, row: t.Dict, callbacks: Callbacks, is_async: bool) -> float:
        assert self.llm is not None, "set LLM before use"
        p_value = self._create_context_recall_prompt(row)
        results = await self.llm.generate(
            p_value,
            callbacks=callbacks,
            is_async=is_async,
            n=self.reproducibility,
        )
        results = [results.generations[0][i].text for i in range(self.reproducibility)]

        answers = [
            await _output_parser.aparse(text, p_value, self.llm, self.max_retries)
            for text in results
        ]

        answers = [answer.dicts() for answer in answers if answer is not None]
        if all(answer is None for answer in answers):
            return np.nan

        answers = ensembler.from_discrete(answers, "attributed")
        answers = ContextRecallClassificationAnswers.parse_obj(answers)

        return self._compute_score(answers)

    def adapt(self, language: str, cache_dir: str | None = None) -> None:
        assert self.llm is not None, "set LLM before use"

        logger.info(f"Adapting Context Recall to {language}")
        self.context_recall_prompt = self.context_recall_prompt.adapt(
            language, self.llm, cache_dir
        )

    def save(self, cache_dir: str | None = None) -> None:
        self.context_recall_prompt.save(cache_dir)


context_recall = ContextRecall()
