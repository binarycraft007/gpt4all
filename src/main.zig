const gpt4all = @import("gpt4all.zig");

pub fn main() void {
    _ = gpt4all.load_mpt_model("test.model", 3);
}
