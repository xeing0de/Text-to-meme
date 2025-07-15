import weaviate

client = weaviate.connect_to_local(
    # host="127.0.0.1",
    port=4654,
    # grpc_port=50051,
)

print(client.is_ready())
client.close()
