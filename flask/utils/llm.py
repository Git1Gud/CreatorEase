from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

class ContentReviewAgent:
    """
    Review agent that uses an LLM to check if content is appropriate, reliable, and relevant.
    """
    def __init__(self, api_key: str):
        self.llm = ChatGroq(api_key=api_key, model="llama3-8b-8192", temperature=0.0)

    def review(self, generated: str, segment: str, context: str) -> str:
        review_prompt = PromptTemplate(
            input_variables=["generated", "segment", "context"],
            template=(
                "You are a content moderation and fact-checking assistant.\n"
                "Given the following:\n"
                "Generated Hook: {generated}\n"
                "Current Segment: {segment}\n"
                "Full Context: {context}\n\n"
                "1. Is the generated hook appropriate (not offensive, hateful, or explicit)?\n"
                "2. Is the hook reliable and relevant to the segment and context?\n"
                "Respond with 'APPROVED' if both are true, otherwise respond with 'REJECTED' and a short reason."
            )
        )
        formatted_prompt = review_prompt.format(
            generated=generated,
            segment=segment,
            context=context
        )
        messages = [HumanMessage(content=formatted_prompt)]
        review_result = self.llm.invoke(messages)
        return review_result.content.strip()

class EngagementQuestionGenerator:
    def __init__(self, api_key: str):
        self.llm = ChatGroq(api_key=api_key, model="llama3-8b-8192", temperature=0.7)
        self.reviewer = ContentReviewAgent(api_key=api_key)

    def generate_question(self, segment_text: str, complete_segment: list) -> str:
        complete_segment_text = "\n".join(complete_segment)

        prompt = PromptTemplate(
            input_variables=["segment_text", "complete_segment"],
            template=(
                "You are an expert content creator specializing in viral tech content. "
                "Your task is to create an engaging hook that will make viewers want to watch the full video.\n\n"
                "Rules:\n"
                "1. Create a thought-provoking question or statement\n"
                "2. Add a call-to-action to watch the full video\n"
                "3. Keep it concise and engaging\n"
                "4. Focus on the most interesting aspect of the segment\n\n"
                "Current Segment:\n{segment_text}\n\n"
                "Full Context (for reference only):\n{complete_segment}\n\n"
                "Generate a compelling hook following this format:\n"
                "[Interesting Question/Statement] + Watch the full video to [benefit/value proposition]\n\n"
                "Examples:\n"
                "- 'Could AI replace human creativity? Watch the full video to discover why experts are divided on this controversial topic.'\n"
                "- 'This breakthrough in AR could change everything... Watch the full podcast to learn how it affects your future.'\n\n"
                "Provide only the hook text, no additional formatting or explanation."
            )
        )

        formatted_prompt = prompt.format(
            segment_text=segment_text,
            complete_segment=complete_segment_text
        )

        messages = [HumanMessage(content=formatted_prompt)]
        question = self.llm.invoke(messages)
        generated = question.content

        # Review the generated question using another LLM call
        review_result = self.reviewer.review(generated, segment_text, complete_segment_text)
        if review_result.startswith("APPROVED"):
            return generated
        else:
            return f"Review failed: {review_result}"

# # Example usage
# if __name__ == "__main__":
#     generator = EngagementQuestionGenerator(api_key=os.getenv("GROQ_API_KEY"))

#     # Current segment text
#     segment_text = (
#         "SPEAKER_00 & SPEAKER_01 (0.03-7.90): Well, I'd love to start with these. "
#         "Years of work right there. Someone on your team called these the real-life Tony Stark glasses. "
#         "Very hard to make each one of these."
#     )

#     # Complete podcast context as a list of segments
#     complete_segment = [
#         "SPEAKER_00 & SPEAKER_01 (0.03-7.90): Well, I'd love to start with these. "
#         "Years of work right there. Someone on your team called these the real-life Tony Stark glasses. "
#         "Very hard to make each one of these.",
#         "SPEAKER_00 & SPEAKER_01 (7.92-16.68): That makes me feel incredibly optimistic. "
#         "In a world where AI gets smarter and smarter. This is probably going to be the next major platform after phones.",
#         "SPEAKER_00 & SPEAKER_01 (13.90-21.95): I miss hugging my mom. Yeah. Haptics is hard. "
#         "How does generative AI change how social media feels? We haven't found the end yet.",
#         "SPEAKER_00 & SPEAKER_01 (16.90-32.68): How does generative AI change how social media feels? "
#         "We haven't found the end yet. The average American has fewer friends now than they did years ago. "
#         "Why do you think that's happening? I mean, there's a lot going on to unpack there."
#     ]

#     # Generate the question
#     question = generator.generate_question(segment_text, complete_segment)
#     print("Generated Question:", question)