use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::model::{
    Bsor, ControllerOffsets, Cut, Frame, Height, Info, Note, Pause, UserData, VrObject, Wall,
    MAGIC_NUMBER,
};

fn read_string<R: Read>(reader: &mut R) -> Result<String> {
    let length = reader.read_u32::<LittleEndian>()? as usize;
    if length == 0 {
        return Ok(String::new());
    }
    let mut buffer = vec![0_u8; length];
    reader.read_exact(&mut buffer)?;
    String::from_utf8(buffer).context("invalid UTF-8 string")
}

fn read_string_maybe_utf16<R: Read + Seek>(reader: &mut R) -> Result<String> {
    let length = reader.read_u32::<LittleEndian>()? as usize;
    if length == 0 {
        return Ok(String::new());
    }

    let mut buffer = vec![0_u8; length];
    reader.read_exact(&mut buffer)?;

    loop {
        let marker_pos = reader.stream_position()?;
        match reader.read_u32::<LittleEndian>() {
            Ok(next_len) if next_len <= 100 => {
                reader.seek(SeekFrom::Start(marker_pos))?;
                break;
            }
            Ok(_) => {
                reader.seek(SeekFrom::Start(marker_pos))?;
                let byte = reader.read_u8()?;
                buffer.push(byte);
            }
            Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => {
                reader.seek(SeekFrom::Start(marker_pos))?;
                break;
            }
            Err(err) => return Err(err.into()),
        }
    }

    String::from_utf8(buffer).context("invalid UTF-8 string in maybe-utf16 field")
}

fn write_string<W: Write>(writer: &mut W, value: &str) -> Result<()> {
    let bytes = value.as_bytes();
    writer.write_u32::<LittleEndian>(bytes.len() as u32)?;
    writer.write_all(bytes)?;
    Ok(())
}

fn peek_u8<R: Read + Seek>(reader: &mut R) -> Result<Option<u8>> {
    let pos = reader.stream_position()?;
    let mut buf = [0_u8; 1];
    match reader.read(&mut buf)? {
        0 => {
            reader.seek(SeekFrom::Start(pos))?;
            Ok(None)
        }
        1 => {
            reader.seek(SeekFrom::Start(pos))?;
            Ok(Some(buf[0]))
        }
        _ => unreachable!(),
    }
}

fn expect_section_magic<R: Read>(reader: &mut R, expected: u8, section: &str) -> Result<()> {
    let actual = reader.read_u8()?;
    if actual != expected {
        bail!("{section} magic number must be {expected}, got {actual}");
    }
    Ok(())
}

fn read_vr_object<R: Read>(reader: &mut R) -> Result<VrObject> {
    Ok(VrObject {
        x: reader.read_f32::<LittleEndian>()?,
        y: reader.read_f32::<LittleEndian>()?,
        z: reader.read_f32::<LittleEndian>()?,
        x_rot: reader.read_f32::<LittleEndian>()?,
        y_rot: reader.read_f32::<LittleEndian>()?,
        z_rot: reader.read_f32::<LittleEndian>()?,
        w_rot: reader.read_f32::<LittleEndian>()?,
    })
}

fn write_vr_object<W: Write>(writer: &mut W, value: &VrObject) -> Result<()> {
    writer.write_f32::<LittleEndian>(value.x)?;
    writer.write_f32::<LittleEndian>(value.y)?;
    writer.write_f32::<LittleEndian>(value.z)?;
    writer.write_f32::<LittleEndian>(value.x_rot)?;
    writer.write_f32::<LittleEndian>(value.y_rot)?;
    writer.write_f32::<LittleEndian>(value.z_rot)?;
    writer.write_f32::<LittleEndian>(value.w_rot)?;
    Ok(())
}

fn read_info<R: Read + Seek>(reader: &mut R) -> Result<Info> {
    expect_section_magic(reader, 0, "info")?;
    Ok(Info {
        version: read_string(reader)?,
        game_version: read_string(reader)?,
        timestamp: read_string(reader)?,
        player_id: read_string(reader)?,
        player_name: read_string_maybe_utf16(reader)?,
        platform: read_string(reader)?,
        tracking_system: read_string(reader)?,
        hmd: read_string(reader)?,
        controller: read_string(reader)?,
        song_hash: read_string(reader)?,
        song_name: read_string_maybe_utf16(reader)?,
        mapper: read_string_maybe_utf16(reader)?,
        difficulty: read_string(reader)?,
        score: reader.read_u32::<LittleEndian>()?,
        mode: read_string(reader)?,
        environment: read_string(reader)?,
        modifiers: read_string(reader)?,
        jump_distance: reader.read_f32::<LittleEndian>()?,
        left_handed: reader.read_u8()? == 1,
        height: reader.read_f32::<LittleEndian>()?,
        start_time: reader.read_f32::<LittleEndian>()?,
        fail_time: reader.read_f32::<LittleEndian>()?,
        speed: reader.read_f32::<LittleEndian>()?,
    })
}

fn write_info<W: Write>(writer: &mut W, info: &Info) -> Result<()> {
    writer.write_u8(0)?;
    write_string(writer, &info.version)?;
    write_string(writer, &info.game_version)?;
    write_string(writer, &info.timestamp)?;
    write_string(writer, &info.player_id)?;
    write_string(writer, &info.player_name)?;
    write_string(writer, &info.platform)?;
    write_string(writer, &info.tracking_system)?;
    write_string(writer, &info.hmd)?;
    write_string(writer, &info.controller)?;
    write_string(writer, &info.song_hash)?;
    write_string(writer, &info.song_name)?;
    write_string(writer, &info.mapper)?;
    write_string(writer, &info.difficulty)?;
    writer.write_u32::<LittleEndian>(info.score)?;
    write_string(writer, &info.mode)?;
    write_string(writer, &info.environment)?;
    write_string(writer, &info.modifiers)?;
    writer.write_f32::<LittleEndian>(info.jump_distance)?;
    writer.write_u8(u8::from(info.left_handed))?;
    writer.write_f32::<LittleEndian>(info.height)?;
    writer.write_f32::<LittleEndian>(info.start_time)?;
    writer.write_f32::<LittleEndian>(info.fail_time)?;
    writer.write_f32::<LittleEndian>(info.speed)?;
    Ok(())
}

fn read_frames<R: Read>(reader: &mut R) -> Result<Vec<Frame>> {
    expect_section_magic(reader, 1, "frames")?;
    let count = reader.read_u32::<LittleEndian>()? as usize;
    let mut items = Vec::with_capacity(count);
    for _ in 0..count {
        items.push(Frame {
            time: reader.read_f32::<LittleEndian>()?,
            fps: reader.read_u32::<LittleEndian>()?,
            head: read_vr_object(reader)?,
            left_hand: read_vr_object(reader)?,
            right_hand: read_vr_object(reader)?,
        });
    }
    Ok(items)
}

fn write_frames<W: Write>(writer: &mut W, frames: &[Frame]) -> Result<()> {
    writer.write_u8(1)?;
    writer.write_u32::<LittleEndian>(frames.len() as u32)?;
    for frame in frames {
        writer.write_f32::<LittleEndian>(frame.time)?;
        writer.write_u32::<LittleEndian>(frame.fps)?;
        write_vr_object(writer, &frame.head)?;
        write_vr_object(writer, &frame.left_hand)?;
        write_vr_object(writer, &frame.right_hand)?;
    }
    Ok(())
}

fn read_cut<R: Read>(reader: &mut R) -> Result<Cut> {
    Ok(Cut {
        speed_ok: reader.read_u8()? == 1,
        direction_ok: reader.read_u8()? == 1,
        saber_type_ok: reader.read_u8()? == 1,
        was_cut_too_soon: reader.read_u8()? == 1,
        saber_speed: reader.read_f32::<LittleEndian>()?,
        saber_direction: [
            reader.read_f32::<LittleEndian>()?,
            reader.read_f32::<LittleEndian>()?,
            reader.read_f32::<LittleEndian>()?,
        ],
        saber_type: reader.read_u32::<LittleEndian>()?,
        time_deviation: reader.read_f32::<LittleEndian>()?,
        cut_deviation: reader.read_f32::<LittleEndian>()?,
        cut_point: [
            reader.read_f32::<LittleEndian>()?,
            reader.read_f32::<LittleEndian>()?,
            reader.read_f32::<LittleEndian>()?,
        ],
        cut_normal: [
            reader.read_f32::<LittleEndian>()?,
            reader.read_f32::<LittleEndian>()?,
            reader.read_f32::<LittleEndian>()?,
        ],
        cut_distance_to_center: reader.read_f32::<LittleEndian>()?,
        cut_angle: reader.read_f32::<LittleEndian>()?,
        before_cut_rating: reader.read_f32::<LittleEndian>()?,
        after_cut_rating: reader.read_f32::<LittleEndian>()?,
    })
}

fn write_cut<W: Write>(writer: &mut W, cut: &Cut) -> Result<()> {
    writer.write_u8(u8::from(cut.speed_ok))?;
    writer.write_u8(u8::from(cut.direction_ok))?;
    writer.write_u8(u8::from(cut.saber_type_ok))?;
    writer.write_u8(u8::from(cut.was_cut_too_soon))?;
    writer.write_f32::<LittleEndian>(cut.saber_speed)?;
    for value in cut.saber_direction {
        writer.write_f32::<LittleEndian>(value)?;
    }
    writer.write_u32::<LittleEndian>(cut.saber_type)?;
    writer.write_f32::<LittleEndian>(cut.time_deviation)?;
    writer.write_f32::<LittleEndian>(cut.cut_deviation)?;
    for value in cut.cut_point {
        writer.write_f32::<LittleEndian>(value)?;
    }
    for value in cut.cut_normal {
        writer.write_f32::<LittleEndian>(value)?;
    }
    writer.write_f32::<LittleEndian>(cut.cut_distance_to_center)?;
    writer.write_f32::<LittleEndian>(cut.cut_angle)?;
    writer.write_f32::<LittleEndian>(cut.before_cut_rating)?;
    writer.write_f32::<LittleEndian>(cut.after_cut_rating)?;
    Ok(())
}

fn read_notes<R: Read>(reader: &mut R) -> Result<Vec<Note>> {
    expect_section_magic(reader, 2, "notes")?;
    let count = reader.read_u32::<LittleEndian>()? as usize;
    let mut items = Vec::with_capacity(count);
    for _ in 0..count {
        let mut note = Note {
            note_id: reader.read_u32::<LittleEndian>()?,
            scoring_type: 0,
            line_index: 0,
            note_line_layer: 0,
            color_type: 0,
            cut_direction: 0,
            event_time: reader.read_f32::<LittleEndian>()?,
            spawn_time: reader.read_f32::<LittleEndian>()?,
            event_type: reader.read_u32::<LittleEndian>()?,
            cut: None,
            pre_score: 0,
            post_score: 0,
            acc_score: 0,
            score: 0,
        };
        if note.event_type == crate::model::NOTE_EVENT_GOOD
            || note.event_type == crate::model::NOTE_EVENT_BAD
        {
            note.cut = Some(read_cut(reader)?);
        }
        note.refresh_derived_fields();
        items.push(note);
    }
    Ok(items)
}

fn write_notes<W: Write>(writer: &mut W, notes: &[Note]) -> Result<()> {
    writer.write_u8(2)?;
    writer.write_u32::<LittleEndian>(notes.len() as u32)?;
    for note in notes {
        writer.write_u32::<LittleEndian>(note.note_id)?;
        writer.write_f32::<LittleEndian>(note.event_time)?;
        writer.write_f32::<LittleEndian>(note.spawn_time)?;
        writer.write_u32::<LittleEndian>(note.event_type)?;
        if note.event_type == crate::model::NOTE_EVENT_GOOD
            || note.event_type == crate::model::NOTE_EVENT_BAD
        {
            let cut = note
                .cut
                .as_ref()
                .ok_or_else(|| anyhow!("note {} is missing cut data", note.note_id))?;
            write_cut(writer, cut)?;
        }
    }
    Ok(())
}

fn read_walls<R: Read>(reader: &mut R) -> Result<Vec<Wall>> {
    expect_section_magic(reader, 3, "walls")?;
    let count = reader.read_u32::<LittleEndian>()? as usize;
    let mut items = Vec::with_capacity(count);
    for _ in 0..count {
        items.push(Wall {
            id: reader.read_u32::<LittleEndian>()?,
            energy: reader.read_f32::<LittleEndian>()?,
            time: reader.read_f32::<LittleEndian>()?,
            spawn_time: reader.read_f32::<LittleEndian>()?,
        });
    }
    Ok(items)
}

fn write_walls<W: Write>(writer: &mut W, walls: &[Wall]) -> Result<()> {
    writer.write_u8(3)?;
    writer.write_u32::<LittleEndian>(walls.len() as u32)?;
    for wall in walls {
        writer.write_u32::<LittleEndian>(wall.id)?;
        writer.write_f32::<LittleEndian>(wall.energy)?;
        writer.write_f32::<LittleEndian>(wall.time)?;
        writer.write_f32::<LittleEndian>(wall.spawn_time)?;
    }
    Ok(())
}

fn read_heights<R: Read>(reader: &mut R) -> Result<Vec<Height>> {
    expect_section_magic(reader, 4, "heights")?;
    let count = reader.read_u32::<LittleEndian>()? as usize;
    let mut items = Vec::with_capacity(count);
    for _ in 0..count {
        items.push(Height {
            height: reader.read_f32::<LittleEndian>()?,
            time: reader.read_f32::<LittleEndian>()?,
        });
    }
    Ok(items)
}

fn write_heights<W: Write>(writer: &mut W, heights: &[Height]) -> Result<()> {
    writer.write_u8(4)?;
    writer.write_u32::<LittleEndian>(heights.len() as u32)?;
    for height in heights {
        writer.write_f32::<LittleEndian>(height.height)?;
        writer.write_f32::<LittleEndian>(height.time)?;
    }
    Ok(())
}

fn read_pauses<R: Read>(reader: &mut R) -> Result<Vec<Pause>> {
    expect_section_magic(reader, 5, "pauses")?;
    let count = reader.read_u32::<LittleEndian>()? as usize;
    let mut items = Vec::with_capacity(count);
    for _ in 0..count {
        items.push(Pause {
            duration: reader.read_u64::<LittleEndian>()?,
            time: reader.read_f32::<LittleEndian>()?,
        });
    }
    Ok(items)
}

fn write_pauses<W: Write>(writer: &mut W, pauses: &[Pause]) -> Result<()> {
    writer.write_u8(5)?;
    writer.write_u32::<LittleEndian>(pauses.len() as u32)?;
    for pause in pauses {
        writer.write_u64::<LittleEndian>(pause.duration)?;
        writer.write_f32::<LittleEndian>(pause.time)?;
    }
    Ok(())
}

fn read_controller_offsets<R: Read>(reader: &mut R) -> Result<ControllerOffsets> {
    Ok(ControllerOffsets {
        left: read_vr_object(reader)?,
        right: read_vr_object(reader)?,
    })
}

fn write_controller_offsets<W: Write>(writer: &mut W, offsets: &ControllerOffsets) -> Result<()> {
    writer.write_u8(6)?;
    write_vr_object(writer, &offsets.left)?;
    write_vr_object(writer, &offsets.right)?;
    Ok(())
}

fn read_user_data_entry<R: Read>(reader: &mut R) -> Result<UserData> {
    let key = read_string(reader)?;
    let byte_count = reader.read_u32::<LittleEndian>()? as usize;
    let mut bytes = vec![0_u8; byte_count];
    reader.read_exact(&mut bytes)?;
    Ok(UserData { key, bytes })
}

fn read_user_data<R: Read>(reader: &mut R) -> Result<Vec<UserData>> {
    let count = reader.read_u32::<LittleEndian>()? as usize;
    let mut items = Vec::with_capacity(count);
    for _ in 0..count {
        items.push(read_user_data_entry(reader)?);
    }
    Ok(items)
}

fn write_user_data<W: Write>(writer: &mut W, user_data: &[UserData]) -> Result<()> {
    writer.write_u8(7)?;
    writer.write_u32::<LittleEndian>(user_data.len() as u32)?;
    for entry in user_data {
        write_string(writer, &entry.key)?;
        writer.write_u32::<LittleEndian>(entry.bytes.len() as u32)?;
        writer.write_all(&entry.bytes)?;
    }
    Ok(())
}

pub fn read_bsor<R: Read + Seek>(reader: &mut R) -> Result<Bsor> {
    let magic_number = reader.read_u32::<LittleEndian>()?;
    if magic_number != MAGIC_NUMBER {
        bail!(
            "file magic number must be {:#x}, got {:#x}",
            MAGIC_NUMBER,
            magic_number
        );
    }

    let file_version = reader.read_u8()?;
    let info = read_info(reader)?;
    let frames = read_frames(reader)?;
    let notes = read_notes(reader)?;
    let walls = read_walls(reader)?;
    let heights = read_heights(reader)?;
    let pauses = read_pauses(reader)?;

    let mut controller_offsets = None;
    let mut user_data = Vec::new();

    if let Some(section_magic) = peek_u8(reader)? {
        match section_magic {
            6 => {
                reader.read_u8()?;
                controller_offsets = Some(read_controller_offsets(reader)?);
                if let Some(next_magic) = peek_u8(reader)? {
                    if next_magic == 7 {
                        reader.read_u8()?;
                        user_data = read_user_data(reader)?;
                    } else {
                        bail!("unexpected trailing section magic {next_magic} after controller offsets");
                    }
                }
            }
            7 => {
                reader.read_u8()?;
                user_data = read_user_data(reader)?;
            }
            other => bail!("unexpected trailing section magic {other}"),
        }
    }

    Ok(Bsor {
        magic_number,
        file_version,
        info,
        frames,
        notes,
        walls,
        heights,
        pauses,
        controller_offsets,
        user_data,
    })
}

pub fn write_bsor<W: Write>(writer: &mut W, replay: &Bsor) -> Result<()> {
    writer.write_u32::<LittleEndian>(replay.magic_number)?;
    writer.write_u8(replay.file_version)?;
    write_info(writer, &replay.info)?;
    write_frames(writer, &replay.frames)?;
    write_notes(writer, &replay.notes)?;
    write_walls(writer, &replay.walls)?;
    write_heights(writer, &replay.heights)?;
    write_pauses(writer, &replay.pauses)?;
    if let Some(offsets) = &replay.controller_offsets {
        write_controller_offsets(writer, offsets)?;
    }
    if !replay.user_data.is_empty() {
        write_user_data(writer, &replay.user_data)?;
    }
    Ok(())
}

pub fn read_bsor_path(path: &Path) -> Result<Bsor> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut reader = BufReader::new(file);
    read_bsor(&mut reader)
}

pub fn write_bsor_path(path: &Path, replay: &Bsor) -> Result<()> {
    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    write_bsor(&mut writer, replay)
}
