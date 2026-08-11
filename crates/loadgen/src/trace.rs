use anyhow::Result;
use serde::Deserialize;

#[derive(Deserialize)]
struct RawConversation {
    #[serde(default)]
    id: String,
    conversations: Vec<RawTurn>,
}

#[derive(Deserialize)]
struct RawTurn {
    from: String,
    value: String,
}

pub struct Conversation {
    pub id: String,
    /// turns[k] = full prompt for the k-th request (all history + k-th human msg)
    pub turns: Vec<String>,
}

pub fn load_sharegpt(path: &str, max_conversations: usize) -> Result<Vec<Conversation>> {
    let raw: Vec<RawConversation> =
        serde_json::from_reader(std::io::BufReader::new(std::fs::File::open(path)?))?;
    let mut out = Vec::new();
    for rc in raw {
        let mut turns = Vec::new();
        let mut history = String::new();
        for t in &rc.conversations {
            if t.from == "human" {
                history.push_str("USER: ");
                history.push_str(&t.value);
                history.push('\n');
                turns.push(history.clone());
            } else {
                history.push_str("ASSISTANT: ");
                history.push_str(&t.value);
                history.push('\n');
            }
        }
        if !turns.is_empty() {
            out.push(Conversation { id: rc.id, turns });
        }
        if out.len() >= max_conversations {
            break;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sharegpt_small.json"
    );

    #[test]
    fn parses_and_builds_growing_prompts() {
        let convs = load_sharegpt(FIXTURE, 10).unwrap();
        assert_eq!(convs.len(), 2);
        let turns = &convs[0].turns;
        assert_eq!(turns.len(), 2); // two human turns
        assert!(turns[0].contains("What is Rust?"));
        assert!(turns[1].starts_with(&turns[0][..])); // growing prefix
        assert!(turns[1].contains("Why is it fast?"));
    }

    #[test]
    fn cap_limits_conversations() {
        let convs = load_sharegpt(FIXTURE, 1).unwrap();
        assert_eq!(convs.len(), 1);
    }
}
