pub mod blocks;

pub mod pb {
    tonic::include_proto!("inference");
}

pub use pb::grpc_inference_service_client::GrpcInferenceServiceClient;
pub use pb::grpc_inference_service_server::{GrpcInferenceService, GrpcInferenceServiceServer};

#[cfg(test)]
mod tests {
    use super::pb;

    #[test]
    fn proto_types_roundtrip() {
        let req = pb::ModelInferRequest {
            model_name: "m".into(),
            inputs: vec![pb::model_infer_request::InferInputTensor {
                name: "text_input".into(),
                datatype: "BYTES".into(),
                shape: vec![1],
                contents: Some(pb::InferTensorContents {
                    bytes_contents: vec![b"hello".to_vec()],
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        };
        assert_eq!(
            req.inputs[0].contents.as_ref().unwrap().bytes_contents[0],
            b"hello"
        );
    }
}
