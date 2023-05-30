const std = @import("std");
const Model = @import("Model.zig");

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(
        std.heap.page_allocator,
    );
    defer arena.deinit();

    var model = try Model.init(.{
        .mtype = .gptj,
        .path = "test.model",
        .threads = 3,
        .response_fn = responseCallback,
    });
    defer model.deinit();

    while (true) {
        // TODO get user input
        var result = try model.predict(.{
            .text = "123456",
            .allocator = arena.allocator(),
        });
        defer arena.allocator().free(result);
    }
}

fn responseCallback(token_id: i32, response: [*c]const u8) callconv(.C) bool {
    _ = token_id;
    _ = response;
    return true;
}
