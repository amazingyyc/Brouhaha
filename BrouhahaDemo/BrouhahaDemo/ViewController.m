#import "ViewController.h"
#import "LeNetViewController.h"
#import "ArtTransformViewController.h"
#import "ArtTransformHalfViewController.h"

@interface ViewController () <UITableViewDataSource, UITableViewDelegate>

@property (nonatomic, strong) UITableView *tableView;

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    
    CGRect bounds = self.view.bounds;
    
    _tableView = [[UITableView alloc] initWithFrame:CGRectMake(0, 20, bounds.size.width, bounds.size.height - 20)
                                              style:UITableViewStylePlain];
    _tableView.dataSource = self;
    _tableView.delegate   = self;
    
    [_tableView registerClass:[UITableViewCell class] forCellReuseIdentifier:@"cellId"];
    
    [self.view addSubview:_tableView];
    [self.view setBackgroundColor:[UIColor whiteColor]];
}

- (NSInteger)tableView:(UITableView *)tableView numberOfRowsInSection:(NSInteger)section {
    if (0 == section) {
        return 3;
    }
    
    return 0;
}

- (UITableViewCell *)tableView:(UITableView *)tableView cellForRowAtIndexPath:(NSIndexPath *)indexPath {
    UITableViewCell *cell = [_tableView dequeueReusableCellWithIdentifier:@"cellId"];
    
    if (0 == indexPath.row) {
        cell.textLabel.text = @"LeNet Float32";
    } else if (1 == indexPath.row) {
        cell.textLabel.text = @"Artistic Style Transform Float32";
    } else if (2 == indexPath.row) {
        cell.textLabel.text = @"Artistic Style Transform Float16";
    }
    
    return cell;
}

- (CGFloat)tableView:(UITableView *)tableView heightForRowAtIndexPath:(NSIndexPath *)indexPath {
    return 100;
}

- (void)tableView:(UITableView *)tableView didSelectRowAtIndexPath:(NSIndexPath *)indexPath {
    [tableView deselectRowAtIndexPath:indexPath animated:YES];
    
    if (0 == indexPath.row) {
        LeNetViewController *ctrl = [[LeNetViewController alloc] init];
        
        [self presentViewController:ctrl animated:YES completion:nil];
    } else if (1 == indexPath.row) {
        ArtTransformViewController *ctrl = [[ArtTransformViewController alloc] init];
        
        [self presentViewController:ctrl animated:YES completion:nil];
    } else if (2 == indexPath.row) {
        ArtTransformHalfViewController *ctrl = [[ArtTransformHalfViewController alloc] init];
        
        [self presentViewController:ctrl animated:YES completion:nil];
    }
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
}

@end











