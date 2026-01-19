from dataclasses import dataclass
from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.pub import DataWriter
from cyclonedds.sub import DataReader
from cyclonedds.idl import IdlStruct

@dataclass
class TestMsg(IdlStruct):
    x: int
    y: int

try:
    # Create a participant
    participant = DomainParticipant(0)
    print("Participant created:", participant)

    # Create a topic
    topic = Topic(participant, "TestTopic", TestMsg)
    print("Topic created:", topic)

    # Create writer and reader
    writer = DataWriter(participant, topic)
    reader = DataReader(participant, topic)

    # Write and read a sample
    writer.write(TestMsg(x=1, y=2))
    sample = reader.take_one(timeout=1.0)
    print("Received:", sample)

except Exception as e:
    print("CycloneDDS test failed:", e)
