#import "PaintView.h"

static const CGFloat RADIUS = 40;

@interface PaintView()

@property(nonatomic, strong) NSMutableArray *points;

@end

@implementation PaintView

- (instancetype)initWithFrame:(CGRect)frame {
    self = [super initWithFrame:frame];
    
    if (self) {
        self.backgroundColor = [UIColor whiteColor];
        _points = [[NSMutableArray alloc] init];
    }
    
    return self;
}

- (void)touchesBegan:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event {
    NSMutableArray *path = [[NSMutableArray alloc] init];
    
    CGPoint p = [[touches anyObject]locationInView:self];
    [path addObject:[NSValue valueWithCGPoint:p]];
    
    [_points addObject:path];
    
    [self setNeedsDisplay];
}

- (void)touchesMoved:(NSSet<UITouch *> *)touches withEvent:(nullable UIEvent *)event {
    NSMutableArray *path = [_points lastObject];
    
    CGPoint p = [[touches anyObject]locationInView:self];
    [path addObject:[NSValue valueWithCGPoint:p]];
    
    [self setNeedsDisplay];
}

- (void)touchesEnded:(NSSet<UITouch *> *)touches withEvent:(nullable UIEvent *)event {
}

- (void)clear {
    [_points removeAllObjects];
    
    [self setNeedsDisplay];
}

- (void)drawRect:(CGRect)rect {
    [super drawRect:rect];
    
    CGContextRef ctx = UIGraphicsGetCurrentContext();
    
    [[UIColor blackColor] set];
    CGContextSetLineCap(ctx, kCGLineCapRound);
    CGContextSetLineWidth(ctx, RADIUS);
    
    for (NSMutableArray *path in _points) {
        CGMutablePathRef pathRef = CGPathCreateMutable();
        
        for (int i = 0; i < path.count; ++i) {
            CGPoint p = [path[i] CGPointValue];
            
            if (0 == i) {
                CGPathMoveToPoint(pathRef, &CGAffineTransformIdentity, p.x, p.y);
            } else {
                CGPathAddLineToPoint(pathRef, &CGAffineTransformIdentity, p.x, p.y);
            }
        }
        
        CGContextAddPath(ctx, pathRef);
        CGContextStrokePath(ctx);
    }
}

@end
