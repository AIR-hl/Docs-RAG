from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from http import HTTPStatus
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional, Union,
)

import requests
from dashscope.api_entities.dashscope_response import DictMixin
from dashscope.common.error import ModelRequired
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict, model_validator
from requests.exceptions import HTTPError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)
BATCH_SIZE=10

def _create_retry_decorator(embeddings: InfiniEmbeddings) -> Callable[[Any], Any]:
    multiplier = 1
    min_seconds = 1
    max_seconds = 4
    # Wait 2^x * 1 second between each retry starting with
    # 1 seconds, then up to 4 seconds, then 4 seconds afterwards
    return retry(
        reraise=True,
        stop=stop_after_attempt(embeddings.max_retries),
        wait=wait_exponential(multiplier, min=min_seconds, max=max_seconds),
        retry=(retry_if_exception_type(HTTPError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def embed_with_retry(embeddings: InfiniEmbeddings, **kwargs: Any) -> Any:
    """Use tenacity to retry the embedding call."""
    retry_decorator = _create_retry_decorator(embeddings)

    @retry_decorator
    def _embed_with_retry(**kwargs: Any) -> Any:
        result = []
        i = 0
        input_data = kwargs["input"]
        input_len = len(input_data) if isinstance(input_data, list) else 1
        batch_size = BATCH_SIZE
        while i < input_len:
            kwargs["input"] = (
                input_data[i : i + batch_size]
                if isinstance(input_data, list)
                else input_data
            )
            resp = embeddings.client.call(**kwargs)
            if resp.status_code == 200:
                result += resp.output["embeddings"]
            elif resp.status_code in [400, 401]:
                raise ValueError(
                    f"status_code: {resp.status_code} \n "
                    f"code: {resp.code} \n message: {resp.message}"
                )
            else:
                raise HTTPError(
                    f"HTTP error occurred: status_code: {resp.status_code} \n "
                    f"code: {resp.code} \n message: {resp.message}",
                    response=resp,
                )
            i += batch_size
        return result

    return _embed_with_retry(**kwargs)


class InfiniEmbeddings(BaseModel, Embeddings):

    client: Any = None  #: :meta private:
    """The DashScope client."""
    model: str = "text-embedding-v1"
    dashscope_api_key: Optional[str] = None
    max_retries: int = 5
    """Maximum number of retries to make when generating."""

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        try:
            values["client"] = TextEmbedding
        except ImportError:
            raise ImportError(
                "Could not import dashscope python package. "
                "Please install it with `pip install dashscope`."
            )
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to DashScope's embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = embed_with_retry(
            self, input=texts, text_type="document", model=self.model
        )
        embedding_list = [item["embedding"] for item in embeddings]
        return embedding_list

    def embed_query(self, text: str) -> List[float]:
        """Call out to DashScope's embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        embedding = embed_with_retry(
            self, input=text, text_type="query", model=self.model
        )[0]["embedding"]
        return embedding


class BaseApi:
    """BaseApi, internal use only.

    """
    @classmethod
    def _validate_params(cls, api_key, model):
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if model is None or not model:
            raise ModelRequired('Model is required!')
        return api_key, model

    @classmethod
    def call(cls,
             model: str,
             input: object,
             api_key: str,
             url: str):
        api_key, model = BaseApi._validate_params(api_key, model)

        payload = {
            "model": model,
            "input": input['texts']
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        response = _handle_request(url, payload, headers)
        try:
            output=next(response)
        except StopIteration:
            pass
        return output


class TextEmbedding(BaseApi):
    @classmethod
    def call(cls,
             model: str,
             input: Union[str, List[str]],
             **kwargs):
        embedding_input = {}
        if isinstance(input, str):
            embedding_input["texts"] = [input]
        else:
            embedding_input["texts"] = input
        kwargs.pop('stream', False)  # not support streaming output.
        url=os.getenv("OPENAI_BASE_URL")
        api_key=os.getenv("OPENAI_API_KEY")
        return super().call(model=model,
                            input=embedding_input,
                            api_key=api_key,
                            url=url)

@dataclass(init=False)
class InfiniAPIResponse(DictMixin):
    status_code: int
    output: Any
    usage: Any
    request_id: str
    code: str
    message: str

    def __init__(self,
                 status_code: int,
                 output: Any = None,
                 usage: Any = None,
                 request_id: str = '',
                 code: str = '',
                 message: str = '',
                 **kwargs):
        super().__init__(status_code=status_code,
                         output=output,
                         usage=usage,
                         request_id=request_id,
                         code=code,
                         message=message,
                         **kwargs)
def _handle_request(url, payload, headers):
    try:
        if not url.endswith('/'):
            url = url + '/'
        else:
            url = url
        response=requests.post(url+"embeddings", json=payload, headers=headers)
        for rsp in _handle_response(response):
            yield rsp
    except BaseException as e:
        logger.error(e)
        raise e

def _handle_response(response: requests.Response):
    request_id = ''
    if response.status_code == HTTPStatus.OK:
            json_content = response.json()
            logger.debug('Response: %s' % json_content)
            output = None
            usage = None
            if 'data' in json_content:
                output = {'embeddings':json_content['data']}
            if 'usage' in json_content:
                usage = json_content['usage']
            if 'request_id' in json_content:
                request_id = json_content['request_id']
            yield InfiniAPIResponse(request_id=request_id,
                                   status_code=HTTPStatus.OK,
                                   output=output,
                                   usage=usage)
    else:
        error = response.json()
        if 'message' in error:
            msg = error['message']
        if 'msg' in error:
            msg = error['msg']
        if 'code' in error:
            code = error['code']
        if 'request_id' in error:
            request_id = error['request_id']
        yield InfiniAPIResponse(request_id=request_id,
                                status_code=response.status_code,
                                code=code,
                                message=msg)