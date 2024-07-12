from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from langchain_core.pydantic_v1 import BaseModel

from ragas.llms.output_parser import RagasoutputParser, get_json_format_instructions
from ragas.llms.prompt import Prompt, PromptValue
from ragas.metrics._answer_similarity import AnswerSimilarity
from ragas.metrics._faithfulness import (
    LONG_FORM_ANSWER_PROMPT,
    HasSegmentMethod,
    _statements_output_parser,
)
from ragas.metrics.base import (
    EvaluationMode,
    MetricWithEmbeddings,
    MetricWithLLM,
    get_segmenter,
)
from ragas.run_config import RunConfig

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks


logger = logging.getLogger(__name__)


class AnswerCorrectnessClassification(BaseModel):
    TP: t.List[t.Dict[str, t.Any]]
    FP: t.List[t.Dict[str, t.Any]]
    FN: t.List[t.Dict[str, t.Any]]


_output_instructions = get_json_format_instructions(AnswerCorrectnessClassification)
_output_parser = RagasoutputParser(pydantic_object=AnswerCorrectnessClassification)

# CORRECTNESS_INSTRUCTIONS = """\
# Given a ground truth and an answer statements, analyze each statement and classify them in one of the following categories:
#
# - TP (true positive): statements that are present in answer that are also directly supported by the one or more statements in ground truth,
# - FP (false positive): statements present in the answer but not directly supported by any statement in ground truth,
# - FN (false negative): statements found in the ground truth but not present in answer.
#
# Each statement can only belong to one of the categories. Provide a reason for each classification.
# """
CORRECTNESS_INSTRUCTIONS = """\
给定一个事实真相（ground truth）和一个回答陈述（answer statements），分析每个陈述并将其分类到以下类别之一：

- TP（真实阳性）：在回答中出现的陈述，这些陈述也直接被一个或多个事实真相中的陈述所支持，
- FP（虚假阳性）：出现在回答中的陈述，但没有被事实真相中的任何陈述直接支持，
- FN（虚假阴性）：在事实真相中找到的陈述，但在回答中并未出现。
每个陈述只能属于上述类别的其中一个。为每个分类提供理由。
"""

CORRECTNESS_PROMPT = Prompt(
    name="answer_correctness",
    instruction=CORRECTNESS_INSTRUCTIONS,
    output_format_instruction=_output_instructions,
    # examples=[
    #     {
    #         "question": """What powers the sun and what is its primary function?""",
    #         "answer": [
    #             "The sun is powered by nuclear fission, similar to nuclear reactors on Earth.",
    #             "The primary function of the sun is to provide light to the solar system.",
    #         ],
    #         "ground_truth": [
    #             "The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium.",
    #             "This fusion process in the sun's core releases a tremendous amount of energy.",
    #             "The energy from the sun provides heat and light, which are essential for life on Earth.",
    #             "The sun's light plays a critical role in Earth's climate system.",
    #             "Sunlight helps to drive the weather and ocean currents.",
    #         ],
    #         "classification": AnswerCorrectnessClassification.parse_obj(
    #             {
    #                 "TP": [
    #                     {
    #                         "statement": "The primary function of the sun is to provide light to the solar system.",
    #                         "reason": "This statement is somewhat supported by the ground truth mentioning the sun providing light and its roles, though it focuses more broadly on the sun's energy.",
    #                     }
    #                 ],
    #                 "FP": [
    #                     {
    #                         "statement": "The sun is powered by nuclear fission, similar to nuclear reactors on Earth.",
    #                         "reason": "This statement is incorrect and contradicts the ground truth which states that the sun is powered by nuclear fusion.",
    #                     }
    #                 ],
    #                 "FN": [
    #                     {
    #                         "statement": "The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium.",
    #                         "reason": "This accurate description of the sun’s power source is not included in the answer.",
    #                     },
    #                     {
    #                         "statement": "This fusion process in the sun's core releases a tremendous amount of energy.",
    #                         "reason": "This process and its significance are not mentioned in the answer.",
    #                     },
    #                     {
    #                         "statement": "The energy from the sun provides heat and light, which are essential for life on Earth.",
    #                         "reason": "The answer only mentions light, omitting the essential aspects of heat and its necessity for life, which the ground truth covers.",
    #                     },
    #                     {
    #                         "statement": "The sun's light plays a critical role in Earth's climate system.",
    #                         "reason": "This broader impact of the sun’s light on Earth's climate system is not addressed in the answer.",
    #                     },
    #                     {
    #                         "statement": "Sunlight helps to drive the weather and ocean currents.",
    #                         "reason": "The effect of sunlight on weather patterns and ocean currents is omitted in the answer.",
    #                     },
    #                 ],
    #             }
    #         ).dict(),
    #     },
    #     {
    #         "question": """What is the boiling point of water?""",
    #         "answer": [
    #             "The boiling point of water is 100 degrees Celsius at sea level"
    #         ],
    #         "ground_truth": [
    #             "The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at sea level.",
    #             "The boiling point of water can change with altitude.",
    #         ],
    #         "classification": AnswerCorrectnessClassification.parse_obj(
    #             {
    #                 "TP": [
    #                     {
    #                         "statement": "The boiling point of water is 100 degrees Celsius at sea level",
    #                         "reason": "This statement is directly supported by the ground truth which specifies the boiling point of water as 100 degrees Celsius at sea level.",
    #                     }
    #                 ],
    #                 "FP": [],
    #                 "FN": [
    #                     {
    #                         "statement": "The boiling point of water can change with altitude.",
    #                         "reason": "This additional information about how the boiling point of water can vary with altitude is not mentioned in the answer.",
    #                     }
    #                 ],
    #             }
    #         ).dict(),
    #     },
    # ],
    examples=[
        # {
        #     "question": """什么驱动着太阳的能量产生？太阳的主要功能是什么？""",
        #     "answer": [
        #         "太阳是由核裂变驱动的，类似于地球上的核反应堆。",
        #         "太阳的主要功能是为太阳系提供光亮。",
        #     ],
        #     "ground_truth": [
        #         "太阳的能量来自核聚变，氢原子聚变形成氦。",
        #         "太阳核心的聚变过程释放出巨大的能量。",
        #         "太阳能提供热量和光，这对地球上的生命至关重要。",
        #         "太阳光在地球气候系统中起着至关重要的作用。",
        #         "阳光有助于推动天气和洋流。",
        #     ],
        #     "classification": AnswerCorrectnessClassification.parse_obj(
        #         {
        #             "TP": [
        #                 {
        #                     "statement": "太阳的主要功能是为太阳系提供光。",
        #                     "reason": "这一说法在一定程度上得到了提到太阳提供光及其作用的基本事实的支持，尽管它更广泛地关注太阳能量。",
        #                 }
        #             ],
        #             "FP": [
        #                 {
        #                     "statement": "太阳由核裂变提供动力，类似于地球上的核反应堆。",
        #                     "reason": "这个说法是错误的，与太阳由核聚变提供动力的基本事实相矛盾。",
        #                 }
        #             ],
        #             "FN": [
        #                 {
        #                     "statement": "太阳由核聚变提供动力，氢原子聚变形成氦。",
        #                     "reason": "答案中没有包括对太阳动力源的准确描述。",
        #                 },
        #                 {
        #                     "statement": "太阳核心的聚变过程释放出巨大的能量。",
        #                     "reason": "答案中没有提到这个过程及其重要性。",
        #                 },
        #                 {
        #                 "statement": "来自太阳的能量提供热量和光，这对地球的生命至关重要",
        #                 "reason": "答案只提到了光，省略了热量的基本方面及其对生命的必要性，而基本事实涵盖了这些方面。",
        #                 },
        #                 {
        #                 "statement": "太阳光在地球气候系统中起着至关重要的作用。",
        #                 "reason": "答案中没有提到太阳光对地球气候系统的这种更广泛的影响。",
        #                 },
        #                 {
        #                 "statement": "阳光有助于推动天气和洋流。",
        #                 "reason": "答案中省略了阳光对天气模式和洋流的影响。"
        #                 },
        #             ],
        #         }
        #     ).dict(),
        # },
        {
            "question": "你是一个经验丰富的中西医全科专家,请根据名词解释的要求及就诊对话内容归纳出病人的病例结果，包括主诉、现病史、既往史信息，然后按指定格式输出。",
            "answer": [
                "主诉：声嘶、头痛、腹泻2天",
                "现病史：患者于2天前外出旅游后发病，主要表现为声音嘶哑、头痛、喉咙痛、咳嗽、浑身无力、发热至36.3℃，伴有腹泻。自服小柴胡颗粒、含片及维生素C，症状部分缓解。失眠，大便次数增多。有冠心病史，长期服用可定。",
                "既往史：\n    1)疾病史：冠心病\n    2)传染病史：无\n    3)手术史：无\n    4)输血史：无\n    5)食物或药物过敏史：无\n    6)预防接种史：无\n    7)家族史：无\n    8)环境及不良习惯：无"
            ],
            "ground_truth": [
                "主诉：咽痛2天",
                "现病史：患者2天前受凉后出现咽痛，乏力，伴全身酸痛、声音嘶哑，有发热，37.7℃，无流涕，无胸闷胸痛，自服药（小柴胡），无好转，纳眠可，二便调，现来就诊。",
                "既往史：既往无特殊，否认高血压、糖尿病等慢性病史；否认肝炎、结核等传染病病史，否认食物、药物过敏史。否认重大外伤、手术史，否认输血史。无吸烟史。无酗酒史。否认家族遗传病史。预防接种不详。"
            ],
            "classification": AnswerCorrectnessClassification.parse_obj(
                {
                    "TP": [
                        {
                            "statement": "主诉中包含声音嘶哑",
                            "reason": "事实真相中的主诉咽痛2天，可以间接关联到声音嘶哑，因为咽痛可能会引起声音嘶哑。"
                        },
                        {
                            "statement": "既往史中无传染病史、手术史、输血史、食物或药物过敏史、预防接种史",
                            "reason": "回答中这些信息与事实真相一致，均未提及相关病史。"
                        }
                    ],
                    "FP": [
                        {
                            "statement": "主诉包含头痛、腹泻",
                            "reason": "这些症状在事实真相的主诉中没有提及，属于额外添加的信息。"
                        },
                        {
                            "statement": "发热至36.3℃",
                            "reason": "事实真相中的发热温度为37.7℃，回答中的温度低于实际，为不正确的信息。"
                        },
                        {
                            "statement": "失眠，大便次数增多",
                            "reason": "这些症状在事实真相中没有提及，为额外添加的细节。"
                        },
                        {
                            "statement": "有冠心病史，长期服用可定",
                            "reason": "事实真相中没有提及冠心病史或长期服用特定药物的信息。"
                        }
                    ],
                    "FN": [
                        {
                            "statement": "现病史中无流涕，无胸闷胸痛",
                            "reason": "回答中没有排除这些阴性症状，而事实真相中有明确的排除说明。"
                        },
                        {
                            "statement": "现病史中自服药无好转",
                            "reason": "回答中提到了症状部分缓解，与事实真相中无好转的信息相矛盾。"
                        },
                        {
                            "statement": "现病史中纳眠可，二便调",
                            "reason": "回答中没有提及食欲和睡眠情况以及大小便情况正常，遗漏了事实真相中的这部分信息。"
                        },
                        {
                            "statement": "既往史中无高血压、糖尿病等慢性病史",
                            "reason": "回答中没有排除这些慢性病史，遗漏了事实真相中的这部分信息。"
                        },
                        {
                            "statement": "既往史中无吸烟史、无酗酒史、否认家族遗传病史",
                            "reason": "回答中没有提及这些历史，遗漏了事实真相中的这部分信息。"
                        }
                    ]
                }
            ).dict(),
        },
        {
            "question": "水的沸点是多少？",
            "answer": [
                "水在海平面的沸点是100摄氏度。"
            ],
            "ground_truth": [
                "水在海平面的沸点是100摄氏度（相当于212华氏度）。",
                "水的沸点会随着海拔高度的变化而变化。"
            ],
            "classification": AnswerCorrectnessClassification.parse_obj(
                {
                  "TP": [
                    {
                      "statement": "水在海平面的沸点是100摄氏度。",
                      "reason": "这一陈述直接由事实支持，事实指出了水在海平面的沸点为100摄氏度。"
                    }
                  ],
                  "FP": [],
                  "FN": [
                    {
                      "statement": "水的沸点会随着海拔高度的变化而变化。",
                      "reason": "关于水的沸点随海拔高度变化的额外信息，在答案中没有提及。"
                    }
                  ]
                }
            ).dict(),
        },
    ],

    input_keys=["question", "answer", "ground_truth"],
    output_key="classification",
    output_type="json",
)


@dataclass
class AnswerCorrectness(MetricWithLLM, MetricWithEmbeddings):

    """
    Measures answer correctness compared to ground truth as a combination of
    factuality and semantic similarity.

    Attributes
    ----------
    name: string
        The name of the metrics
    weights:
        a list of two weights corresponding to factuality and semantic similarity
        Defaults [0.75, 0.25]
    answer_similarity:
        The AnswerSimilarity object
    """

    name: str = "answer_correctness"  # type: ignore[reportIncompatibleMethodOverride]
    evaluation_mode: EvaluationMode = EvaluationMode.qga  # type: ignore[reportIncompatibleMethodOverride]
    correctness_prompt: Prompt = field(default_factory=lambda: CORRECTNESS_PROMPT)
    long_form_answer_prompt: Prompt = field(
        default_factory=lambda: LONG_FORM_ANSWER_PROMPT
    )
    weights: list[float] = field(default_factory=lambda: [0.75, 0.25])
    answer_similarity: t.Optional[AnswerSimilarity] = None
    sentence_segmenter: t.Optional[HasSegmentMethod] = None
    max_retries: int = 1

    def __post_init__(self: t.Self):
        if len(self.weights) != 2:
            raise ValueError(
                "Expects a list of two weights. First for factuality, second for semantic similarity"
            )
        if all([w == 0 for w in self.weights]):
            raise ValueError("At least one weight must be non-zero")
        if not all([w >= 0 for w in self.weights]):
            raise ValueError("Weights must be non-negative")

        if self.sentence_segmenter is None:
            language = self.long_form_answer_prompt.language
            self.sentence_segmenter = get_segmenter(language=language, clean=False)

    def init(self, run_config: RunConfig):
        super().init(run_config)
        if self.answer_similarity is None and self.weights[1] != 0:
            self.answer_similarity = AnswerSimilarity(
                llm=self.llm, embeddings=self.embeddings
            )

    def _compute_statement_presence(
        self, prediction: AnswerCorrectnessClassification
    ) -> float:
        tp = len(prediction.TP)
        fp = len(prediction.FP)
        fn = len(prediction.FN)
        score = tp / (tp + 0.5 * (fp + fn)) if tp > 0 else 0
        return score

    def _create_statements_prompt(self, question: str, text: str) -> PromptValue:
        assert self.sentence_segmenter is not None, "sentence_segmenter is not set"

        sentences = self.sentence_segmenter.segment(text)
        sentences = [
            sentence for sentence in sentences if sentence.strip().endswith(".")
        ]
        sentences = "\n".join([f"{i}:{x}" for i, x in enumerate(sentences)])
        prompt_value = self.long_form_answer_prompt.format(
            question=question, answer=text, sentences=sentences
        )
        return prompt_value

    async def _ascore(self, row: t.Dict, callbacks: Callbacks, is_async: bool) -> float:
        assert self.llm is not None, "LLM must be set"

        question = row["question"]
        statements = {}
        for item in ["answer", "ground_truth"]:
            p_value = self._create_statements_prompt(question, row[item])
            item_statement = await self.llm.generate(
                p_value, callbacks=callbacks, is_async=is_async
            )
            statements[item] = await _statements_output_parser.aparse(
                item_statement.generations[0][0].text,
                p_value,
                self.llm,
                self.max_retries,
            )
            statements[item] = (
                statements[item].dicts() if statements[item] is not None else []
            )

        if not all([val == [] for val in statements.values()]):
            ground_truth = [
                statement
                for item in statements["ground_truth"]
                for statement in item["simpler_statements"]
            ]
            answer = [
                statement
                for item in statements["answer"]
                for statement in item["simpler_statements"]
            ]
            p_value = self.correctness_prompt.format(
                question=question,
                ground_truth=ground_truth,
                answer=answer,
            )
            is_statement_present = await self.llm.generate(
                p_value, callbacks=callbacks, is_async=is_async
            )
            result_text = is_statement_present.generations[0][0].text

            answers = await _output_parser.aparse(
                result_text, p_value, self.llm, self.max_retries
            )
            if answers is None:
                return np.nan

            f1_score = self._compute_statement_presence(answers)
        else:
            f1_score = 1.0

        if self.weights[1] == 0:
            similarity_score = 0.0
        else:
            assert self.answer_similarity is not None, "AnswerSimilarity must be set"

            similarity_score = await self.answer_similarity.ascore(
                row, callbacks=callbacks, is_async=is_async
            )

        score = np.average(
            [f1_score, similarity_score],
            weights=self.weights,
        )

        return float(score)

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        assert self.llm is not None, "llm must be set to compute score"

        logger.info(f"Adapting AnswerCorrectness metric to {language}")
        self.correctness_prompt = self.correctness_prompt.adapt(
            language, self.llm, cache_dir
        )

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        self.correctness_prompt.save(cache_dir)


answer_correctness = AnswerCorrectness()
