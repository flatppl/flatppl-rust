module attributes {flatppl.runtime_status = 1 : i64} {
  func.func @logdensity(%arg0: tensor<2x2xf32>, %arg1: tensor<f32>, %arg2: tensor<2x2xf32>) -> (tensor<f32>, tensor<i32>) {
    %0 = stablehlo.constant dense<0.0> : tensor<f32>
    %1 = stablehlo.compare LE, %arg1, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %2 = stablehlo.constant dense<0> : tensor<i32>
    %3 = stablehlo.constant dense<9> : tensor<i32>
    %4 = stablehlo.select %1, %3, %2 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %5 = stablehlo.cholesky %arg0, lower = true : tensor<2x2xf32>
    %6 = stablehlo.iota dim = 0 : tensor<2x2xf32>
    %7 = stablehlo.iota dim = 1 : tensor<2x2xf32>
    %8 = stablehlo.compare EQ, %6, %7 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
    %9 = stablehlo.constant dense<0.0> : tensor<2x2xf32>
    %10 = stablehlo.select %8, %5, %9 : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.reduce(%10 init: %11) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
    %13 = stablehlo.cholesky %arg2, lower = true : tensor<2x2xf32>
    %14 = stablehlo.log %12 : tensor<2xf32>
    %15 = stablehlo.reduce(%14 init: %11) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %16 = stablehlo.constant dense<2.0> : tensor<f32>
    %17 = stablehlo.multiply %16, %15 : tensor<f32>
    %18 = stablehlo.subtract %17, %17 : tensor<f32>
    %19 = stablehlo.compare EQ, %18, %18 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %20 = stablehlo.select %19, %2, %3 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %21 = stablehlo.compare EQ, %4, %2, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %22 = stablehlo.select %21, %20, %4 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %23 = stablehlo.iota dim = 0 : tensor<2x2xf32>
    %24 = stablehlo.iota dim = 1 : tensor<2x2xf32>
    %25 = stablehlo.compare EQ, %23, %24 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
    %26 = stablehlo.select %25, %13, %9 : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %27 = stablehlo.reduce(%26 init: %11) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
    %28 = stablehlo.log %27 : tensor<2xf32>
    %29 = stablehlo.reduce(%28 init: %11) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %30 = stablehlo.multiply %16, %29 : tensor<f32>
    %31 = "stablehlo.triangular_solve"(%13, %5) <{left_side = true, lower = true, unit_diagonal = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>}> : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %32 = stablehlo.multiply %31, %31 : tensor<2x2xf32>
    %33 = stablehlo.reduce(%32 init: %11) applies stablehlo.add across dimensions = [0] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
    %34 = stablehlo.reduce(%33 init: %11) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %35 = stablehlo.constant dense<0.5> : tensor<f32>
    %36 = stablehlo.multiply %35, %arg1 : tensor<f32>
    %37 = stablehlo.multiply %36, %17 : tensor<f32>
    %38 = stablehlo.constant dense<3.0> : tensor<f32>
    %39 = stablehlo.add %arg1, %38 : tensor<f32>
    %40 = stablehlo.constant dense<-0.5> : tensor<f32>
    %41 = stablehlo.multiply %40, %39 : tensor<f32>
    %42 = stablehlo.multiply %41, %30 : tensor<f32>
    %43 = stablehlo.multiply %40, %34 : tensor<f32>
    %44 = stablehlo.multiply %arg1, %16 : tensor<f32>
    %45 = stablehlo.constant dense<0.6931471805599453> : tensor<f32>
    %46 = stablehlo.multiply %44, %45 : tensor<f32>
    %47 = stablehlo.multiply %40, %46 : tensor<f32>
    %48 = stablehlo.constant dense<0.5723649429247001> : tensor<f32>
    %49 = stablehlo.add %36, %0 : tensor<f32>
    %50 = chlo.lgamma %49 : tensor<f32> -> tensor<f32>
    %51 = stablehlo.add %48, %50 : tensor<f32>
    %52 = stablehlo.add %36, %40 : tensor<f32>
    %53 = chlo.lgamma %52 : tensor<f32> -> tensor<f32>
    %54 = stablehlo.add %51, %53 : tensor<f32>
    %55 = stablehlo.negate %54 : tensor<f32>
    %56 = stablehlo.add %37, %42 : tensor<f32>
    %57 = stablehlo.add %56, %43 : tensor<f32>
    %58 = stablehlo.add %57, %47 : tensor<f32>
    %59 = stablehlo.add %58, %55 : tensor<f32>
    return %59, %22 : tensor<f32>, tensor<i32>
  }
}
