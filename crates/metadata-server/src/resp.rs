//! Minimal Redis RESP2 protocol implementation.
//!
//! Only implements the subset needed by torch's inductor remote cache:
//! PING, GET, SET, EXISTS. That's it. We're not building Redis.
//!
//! Wire format reference: https://redis.io/docs/reference/protocol-spec/

use bytes::{Buf, BytesMut};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RespError {
    #[error("incomplete frame, need more data")]
    Incomplete,
    #[error("protocol error: {0}")]
    Protocol(String),
    #[error("frame too large: {0} bytes")]
    TooLarge(usize),
}

/// 256MB safety limit for RESP bulk string payloads.
const MAX_BULK_LEN: usize = 256 * 1024 * 1024;

#[derive(Debug)]
pub enum RespCommand {
    Ping,
    Get { key: Vec<u8> },
    Set { key: Vec<u8>, value: Vec<u8> },
    Exists { key: Vec<u8> },
    Handshake { command: String },
}

#[derive(Debug)]
pub enum RespResponse {
    SimpleString(&'static str),
    BulkString(Vec<u8>),
    Null,
    Integer(i64),
    Error(String),
    EmptyArray,
}

impl RespResponse {
    pub fn ok() -> Self {
        Self::SimpleString("OK")
    }

    pub fn pong() -> Self {
        Self::SimpleString("PONG")
    }

    pub fn encode(&self, buf: &mut BytesMut) {
        use std::fmt::Write;
        match self {
            Self::SimpleString(s) => {
                let _ = write!(buf, "+{s}\r\n");
            }
            Self::BulkString(data) => {
                let _ = write!(buf, "${}\r\n", data.len());
                buf.extend_from_slice(data);
                buf.extend_from_slice(b"\r\n");
            }
            Self::Null => {
                buf.extend_from_slice(b"$-1\r\n");
            }
            Self::Integer(n) => {
                let _ = write!(buf, ":{n}\r\n");
            }
            Self::Error(msg) => {
                let _ = write!(buf, "-ERR {msg}\r\n");
            }
            Self::EmptyArray => {
                buf.extend_from_slice(b"*0\r\n");
            }
        }
    }
}

pub fn parse_command(buf: &mut BytesMut) -> Result<RespCommand, RespError> {
    let mut cursor = std::io::Cursor::new(&buf[..]);

    let array_len = parse_array_header(&mut cursor)?;
    if array_len == 0 {
        return Err(RespError::Protocol("empty command".into()));
    }

    let mut parts: Vec<Vec<u8>> = Vec::with_capacity(array_len);
    for _ in 0..array_len {
        let part = parse_bulk_string(&mut cursor)?;
        parts.push(part);
    }

    let consumed = cursor.position() as usize;
    buf.advance(consumed);

    let cmd = parts[0].to_ascii_uppercase();
    match cmd.as_slice() {
        b"PING" => Ok(RespCommand::Ping),
        b"GET" => {
            if parts.len() != 2 {
                return Err(RespError::Protocol(
                    "GET requires exactly 1 argument".into(),
                ));
            }
            Ok(RespCommand::Get {
                key: parts.into_iter().nth(1).unwrap(),
            })
        }
        b"SET" => {
            if parts.len() < 3 {
                return Err(RespError::Protocol(
                    "SET requires at least 2 arguments".into(),
                ));
            }
            let mut iter = parts.into_iter();
            iter.next();
            let key = iter.next().unwrap();
            let value = iter.next().unwrap();
            Ok(RespCommand::Set { key, value })
        }
        b"EXISTS" => {
            if parts.len() != 2 {
                return Err(RespError::Protocol(
                    "EXISTS requires exactly 1 argument".into(),
                ));
            }
            Ok(RespCommand::Exists {
                key: parts.into_iter().nth(1).unwrap(),
            })
        }
        b"COMMAND" | b"CONFIG" | b"CLIENT" | b"INFO" => Ok(RespCommand::Handshake {
            command: String::from_utf8_lossy(&cmd).into_owned(),
        }),
        _ => Err(RespError::Protocol(format!(
            "unsupported command: {}",
            String::from_utf8_lossy(&cmd)
        ))),
    }
}

fn parse_array_header(cursor: &mut std::io::Cursor<&[u8]>) -> Result<usize, RespError> {
    let line = read_line(cursor)?;
    if !line.starts_with(b"*") {
        return Err(RespError::Protocol(format!(
            "expected array, got '{}'",
            String::from_utf8_lossy(&line)
        )));
    }
    let n: usize = std::str::from_utf8(&line[1..])
        .map_err(|_| RespError::Protocol("invalid array length".into()))?
        .parse()
        .map_err(|_| RespError::Protocol("invalid array length".into()))?;
    Ok(n)
}

fn parse_bulk_string(cursor: &mut std::io::Cursor<&[u8]>) -> Result<Vec<u8>, RespError> {
    let line = read_line(cursor)?;
    if !line.starts_with(b"$") {
        return Err(RespError::Protocol(format!(
            "expected bulk string, got '{}'",
            String::from_utf8_lossy(&line)
        )));
    }
    let len: usize = std::str::from_utf8(&line[1..])
        .map_err(|_| RespError::Protocol("invalid bulk length".into()))?
        .parse()
        .map_err(|_| RespError::Protocol("invalid bulk length".into()))?;

    if len > MAX_BULK_LEN {
        return Err(RespError::TooLarge(len));
    }

    let pos = cursor.position() as usize;
    let remaining = cursor.get_ref().len() - pos;
    if remaining < len + 2 {
        return Err(RespError::Incomplete);
    }

    let data = cursor.get_ref()[pos..pos + len].to_vec();
    cursor.set_position((pos + len + 2) as u64);
    Ok(data)
}

fn read_line(cursor: &mut std::io::Cursor<&[u8]>) -> Result<Vec<u8>, RespError> {
    let start = cursor.position() as usize;
    let data = cursor.get_ref();

    for i in start..data.len().saturating_sub(1) {
        if data[i] == b'\r' && data[i + 1] == b'\n' {
            let line = data[start..i].to_vec();
            cursor.set_position((i + 2) as u64);
            return Ok(line);
        }
    }

    Err(RespError::Incomplete)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ping() {
        let mut buf = BytesMut::from("*1\r\n$4\r\nPING\r\n");
        let cmd = parse_command(&mut buf).unwrap();
        assert!(matches!(cmd, RespCommand::Ping));
        assert!(buf.is_empty());
    }

    #[test]
    fn parse_get() {
        let mut buf = BytesMut::from("*2\r\n$3\r\nGET\r\n$8\r\ncachekey\r\n");
        let cmd = parse_command(&mut buf).unwrap();
        match cmd {
            RespCommand::Get { key } => assert_eq!(key, b"cachekey"),
            other => panic!("expected Get, got {other:?}"),
        }
    }

    #[test]
    fn parse_set() {
        let mut buf = BytesMut::from("*3\r\n$3\r\nSET\r\n$2\r\nk1\r\n$3\r\nval\r\n");
        let cmd = parse_command(&mut buf).unwrap();
        match cmd {
            RespCommand::Set { key, value } => {
                assert_eq!(key, b"k1");
                assert_eq!(value, b"val");
            }
            other => panic!("expected Set, got {other:?}"),
        }
    }

    #[test]
    fn parse_exists() {
        let mut buf = BytesMut::from("*2\r\n$6\r\nEXISTS\r\n$2\r\nk1\r\n");
        let cmd = parse_command(&mut buf).unwrap();
        assert!(matches!(cmd, RespCommand::Exists { .. }));
    }

    #[test]
    fn encode_responses() {
        let mut buf = BytesMut::new();

        RespResponse::ok().encode(&mut buf);
        assert_eq!(&buf[..], b"+OK\r\n");
        buf.clear();

        RespResponse::pong().encode(&mut buf);
        assert_eq!(&buf[..], b"+PONG\r\n");
        buf.clear();

        RespResponse::Null.encode(&mut buf);
        assert_eq!(&buf[..], b"$-1\r\n");
        buf.clear();

        RespResponse::Integer(1).encode(&mut buf);
        assert_eq!(&buf[..], b":1\r\n");
        buf.clear();

        RespResponse::BulkString(b"hello".to_vec()).encode(&mut buf);
        assert_eq!(&buf[..], b"$5\r\nhello\r\n");
    }

    #[test]
    fn incomplete_returns_err() {
        let mut buf = BytesMut::from("*2\r\n$3\r\nGET\r\n");
        assert!(matches!(
            parse_command(&mut buf),
            Err(RespError::Incomplete)
        ));
    }
}
