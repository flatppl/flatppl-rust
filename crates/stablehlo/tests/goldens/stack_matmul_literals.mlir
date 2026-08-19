module {
  func.func @logdensity() -> tensor<2x2xi32> {
    %0 = stablehlo.constant dense<1> : tensor<i32>
    %1 = stablehlo.constant dense<3> : tensor<i32>
    %2 = stablehlo.constant dense<5> : tensor<i32>
    %3 = stablehlo.reshape %0 : (tensor<i32>) -> tensor<1xi32>
    %4 = stablehlo.reshape %1 : (tensor<i32>) -> tensor<1xi32>
    %5 = stablehlo.reshape %2 : (tensor<i32>) -> tensor<1xi32>
    %6 = stablehlo.concatenate %3, %4, %5, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %7 = stablehlo.constant dense<9> : tensor<i32>
    %8 = stablehlo.constant dense<5> : tensor<i32>
    %9 = stablehlo.constant dense<1> : tensor<i32>
    %10 = stablehlo.reshape %7 : (tensor<i32>) -> tensor<1xi32>
    %11 = stablehlo.reshape %8 : (tensor<i32>) -> tensor<1xi32>
    %12 = stablehlo.reshape %9 : (tensor<i32>) -> tensor<1xi32>
    %13 = stablehlo.concatenate %10, %11, %12, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %14 = stablehlo.reshape %6 : (tensor<3xi32>) -> tensor<1x3xi32>
    %15 = stablehlo.reshape %13 : (tensor<3xi32>) -> tensor<1x3xi32>
    %16 = stablehlo.concatenate %14, %15, dim = 0 : (tensor<1x3xi32>, tensor<1x3xi32>) -> tensor<2x3xi32>
    %17 = stablehlo.constant dense<1> : tensor<i32>
    %18 = stablehlo.constant dense<0> : tensor<i32>
    %19 = stablehlo.reshape %17 : (tensor<i32>) -> tensor<1xi32>
    %20 = stablehlo.reshape %18 : (tensor<i32>) -> tensor<1xi32>
    %21 = stablehlo.concatenate %19, %20, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %22 = stablehlo.constant dense<0> : tensor<i32>
    %23 = stablehlo.constant dense<1> : tensor<i32>
    %24 = stablehlo.reshape %22 : (tensor<i32>) -> tensor<1xi32>
    %25 = stablehlo.reshape %23 : (tensor<i32>) -> tensor<1xi32>
    %26 = stablehlo.concatenate %24, %25, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %27 = stablehlo.constant dense<1> : tensor<i32>
    %28 = stablehlo.constant dense<1> : tensor<i32>
    %29 = stablehlo.reshape %27 : (tensor<i32>) -> tensor<1xi32>
    %30 = stablehlo.reshape %28 : (tensor<i32>) -> tensor<1xi32>
    %31 = stablehlo.concatenate %29, %30, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %32 = stablehlo.reshape %21 : (tensor<2xi32>) -> tensor<1x2xi32>
    %33 = stablehlo.reshape %26 : (tensor<2xi32>) -> tensor<1x2xi32>
    %34 = stablehlo.reshape %31 : (tensor<2xi32>) -> tensor<1x2xi32>
    %35 = stablehlo.concatenate %32, %33, %34, dim = 0 : (tensor<1x2xi32>, tensor<1x2xi32>, tensor<1x2xi32>) -> tensor<3x2xi32>
    %36 = stablehlo.dot_general %16, %35, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x3xi32>, tensor<3x2xi32>) -> tensor<2x2xi32>
    return %36 : tensor<2x2xi32>
  }
}
