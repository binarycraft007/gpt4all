const Model = @import("Model.zig");

pub fn main() !void {
    var model = try Model.init(.{
        .mtype = .gptj,
        .path = "test.model",
        .threads = 3,
        .prompt_fn = promptCallback,
        .response_fn = responseCallback,
        .recalculate_fn = recalculateCallback,
    });
    defer model.deinit();

    while (true) {
        // TODO get user input
        model.predict(.{ .text = "123456" });
    }
}

fn promptCallback(_: i32) callconv(.C) bool {
    return true;
}

fn responseCallback(token_id: i32, response: [*c]const u8) callconv(.C) bool {
    _ = token_id;
    _ = response;
    return true;
}

fn recalculateCallback(is_recalculating: bool) callconv(.C) bool {
    return is_recalculating;
}
