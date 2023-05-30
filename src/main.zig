const Model = @import("Model.zig");

pub fn main() void {
    var model = Model.init(.{
        .mtype = .gptj,
        .path = "test.model",
        .threads = 3,
    });
    defer model.deinit();
}
