const std = @import("std");
const mem = std.mem;
const io = std.io;
const Model = @import("Model.zig");

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    var model = try Model.init(.{
        .mtype = .gptj,
        .path = "ggml-gpt4all-j-v1.3-groovy.bin",
        .threads = 4,
        .response_fn = struct {
            fn responseCallback(resp: []const u8) bool {
                std.debug.print("{s}", .{resp});
                return true;
            }
        }.responseCallback,
    });
    defer model.deinit();

    while (true) {
        var input = std.ArrayList(u8).init(arena.allocator());
        defer input.deinit();

        std.debug.print("> Question:\n", .{});
        var stdin = io.getStdIn().reader();
        while (true) {
            var buf: [4096]u8 = undefined;
            var line = try stdin.readUntilDelimiterOrEof(&buf, '\n');

            if (line) |l| {
                if (mem.trim(u8, l, " ").len == 0) {
                    break;
                }
                try input.appendSlice(l);
                try input.append('\n');
            }
        }

        std.debug.print("> Answer:\n", .{});
        var result = try model.predict(.{
            .text = input.items,
            .allocator = arena.allocator(),
        });
        defer arena.allocator().free(result);
        std.debug.print("\n", .{});
    }
}
