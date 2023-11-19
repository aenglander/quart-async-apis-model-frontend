import asyncio
import multiprocessing as mp
import queue
from asyncio import Future
from typing import Union
from uuid import UUID, uuid4

import quart
from pydantic import BaseModel, Field
from quart_schema import (
    QuartSchema,
    validate_request,
    validate_response,
    Info,
)
from transformers import pipeline, Pipeline, logging

MODEL = "gpt2"  # Can be replaced with any Hugging Face completion model
MODEL_DIRECTORY = "model"  # Location to store the model files
MAX_GENERATION_TOKEN_LENGTH = 256  # Reduce if having resource issues


async def model_response_processor(
    response_queue: mp.Queue,
    request_futures: dict[UUID, Future[str]],
):
    """
    Responder is a loop to pull responses off of the response queue and
    fulfill the corresponding promise.

    :param response_queue: Queue to poll for responses
    :param request_futures: Dictionary mapping of request IDs to Futures
    """
    while True:
        # Process responses from the response queue forever
        try:
            # Since it's a synchronous in an async function, set block to False
            item: tuple[UUID, Union[str, Exception]] = response_queue.get(
                block=False
            )
            # Break the queue item into its parts
            request_id, response = item
            # Get the future for this request from the mapping
            future = request_futures.pop(request_id)
            # Fulfill the future via the appropriate method
            if isinstance(response, Exception):
                future.set_exception(response)
            else:
                future.set_result(response)
        except queue.Empty:
            # No item, so suspend the current task and allow others to run
            await asyncio.sleep(0)


def model_request_processor(
    model: str,
    max_generation_token_length: int,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
):
    """
    Process to manage processing model requests serially
    :param model: The hugging face model identifier
    :param max_generation_token_length: The maximum number of tokens to generate
    :param request_queue: The queue from which to get requests
    :param response_queue: The queue on which to place responses
    """
    # Create a generator to be used for all requests
    generator: Pipeline = pipeline("text-generation", model=model)
    logging.set_verbosity_error()  # Reducing noise in the logs

    # Run a continuous loop to process requests in order
    while True:
        # Get the next item from the queue. It's a subprocess and okay to block
        item: tuple[UUID, str] = request_queue.get()
        # Break the queue item into its parts
        request_id, request = item
        try:
            # Generate a result from the pipeline
            results = generator(
                request,
                max_length=max_generation_token_length,
                num_return_sequences=1,
                num_workers=0,
            )
            # Get the response text from the result
            response: Union[str, Exception] = results[0]["generated_text"]
        except Exception as e:
            # Any errors are sent back and the response
            response = e
        # Put the response on the queue with the request ID
        response_queue.put((request_id, response))


# Queue for incoming requests
model_request_queue: mp.Queue = mp.Queue()
# Queue for outgoing responses
model_response_queue: mp.Queue = mp.Queue()
# Request ID/Future map for identifying the future for a request
request_future_map: dict[UUID, Future[str]] = {}


app = quart.Quart("model")
quart_schema = QuartSchema(
    app,
    swagger_ui_path="/",
    info=Info(
        title="Quart Model API",
        version="1.0.0",
        description=f"Example Quart API to manage requests for the"
        f" {MODEL} model",
    ),
)


@app.while_serving
async def bootstrap():
    """
    Handling model requests needs to be done in a separate process to
    prevent blocking the worker process from handling requests.
    """
    # Pre-download the model on startup to prevent race conditions
    generator: Pipeline = pipeline("text-generation", model=MODEL)
    generator.save_pretrained(MODEL_DIRECTORY)

    p = mp.Process(
        target=model_request_processor,
        args=(
            MODEL,
            MAX_GENERATION_TOKEN_LENGTH,
            model_request_queue,
            model_response_queue,
        ),
        name="Model Request Processor",
    )
    p.start()
    t = asyncio.create_task(
        model_response_processor(model_response_queue, request_future_map)
    )
    yield
    t.cancel("Shutting down the server")
    p.terminate()


class Request(BaseModel):
    """
    Incoming API request
    """

    request: str = Field(
        title="Request text",
        description="The text you wish to process with the model",
        examples=["Because I'm happy!"],
        min_length=1,
        max_length=2048,
    )


class Response(BaseModel):
    """
    Outgoing API response
    """

    response: str = Field(
        title="Response text",
        examples=[
            "Because I'm Happy! Clap along if you feel"
            " like a room without a roof."
        ],
        description="Response text received from the model",
    )


@app.post("/api")
@validate_request(Request)
@validate_response(Response, 200)
async def api(data: Request):
    """
    Model API endpoint
    """
    # Get a future to be fulfilled by other asynchronous processes
    future: Future[str] = asyncio.get_running_loop().create_future()
    # Uniquely identify this request
    request_id = uuid4()
    # Place the future in the map so the response processor can find it
    request_future_map[request_id] = future

    while True:
        # Queue is synchronous, so try and send until it succeeds
        try:
            # Place the request and ID on the queue
            model_request_queue.put((request_id, data.request), block=False)
            # If the queue put is successful, break the loop
            break
        except queue.Full:
            # Suspend the current task and allow others to run
            await asyncio.sleep(0)
    # Wait until the response processor has fulfilled the future
    response = await future
    # Return the API response
    return Response(response=response)


if __name__ == "__main__":
    """Example for running a hypercorn server"""
    from hypercorn import Config
    from hypercorn.asyncio import serve

    config = Config()
    config.bind = ["0.0.0.0:5000"]  # Same as default for quart run
    config.workers = 1  # Always have 1 worker to ensure 1 model instance
    config.accesslog = "-"
    config.errorlog = "-"
    asyncio.run(serve(app, config))
